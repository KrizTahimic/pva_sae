"""
T-Statistic based SAE latent selector for Phase 2.10.

Uses Welch's t-test to identify SAE features that best distinguish between
correct and incorrect Python code solutions. This provides a more statistically
rigorous alternative to Phase 2.5's simple separation scores.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from scipy import stats
from tqdm import tqdm
from datetime import datetime

from common.config import Config
from common.logging import get_logger
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae

# Module-level logger
logger = get_logger("t_statistic_selector", phase="2.10")


class TStatisticSelector:
    """T-Statistic based selector for PVA latent directions."""
    
    def __init__(self, config: Config):
        """Initialize selector with configuration."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Activation directory from Phase 1
        self.activation_dir = Path(config.phase1_output_dir) / "activations"
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
    
    def _get_task_ids(self, directory: Path) -> List[str]:
        """Extract unique task IDs from activation files."""
        task_ids = set()
        for file in directory.glob("*_layer_*.npz"):
            # Extract task_id from filename pattern: {task_id}_layer_{n}.npz
            parts = file.stem.split('_layer_')
            if len(parts) == 2:
                task_ids.add(parts[0])
        return sorted(list(task_ids))
    
    def load_activations_for_layer(
        self, 
        layer_idx: int, 
        category: str
    ) -> Tuple[List[str], torch.Tensor]:
        """Load all activations for a specific layer and category."""
        task_ids = self.correct_task_ids if category == "correct" else self.incorrect_task_ids
        category_dir = self.correct_dir if category == "correct" else self.incorrect_dir
        
        activations = []
        valid_task_ids = []
        
        for task_id in task_ids:
            filepath = category_dir / f"{task_id}_layer_{layer_idx}.npz"
            if filepath.exists():
                # Load activation (simple numpy array)
                data = np.load(filepath)
                # Get the first (and only) array from the npz file
                activation = torch.from_numpy(data[data.files[0]])
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
        correct_features: torch.Tensor,
        incorrect_features: torch.Tensor
    ) -> Dict[str, List[float]]:
        """
        Calculate t-statistics between correct and incorrect code activations.
        
        Uses Welch's t-test which:
        - Handles unequal variances between groups
        - Provides effect size normalized by pooled variance
        - Returns positive values when first group > second group
        
        Args:
            correct_features: Tensor of shape (n_correct_samples, n_features)
            incorrect_features: Tensor of shape (n_incorrect_samples, n_features)
            
        Returns:
            Dict with 't_stats_correct' (correct > incorrect) and 
            't_stats_incorrect' (incorrect > correct) lists
        """
        t_stats_correct = []  # Correct > Incorrect direction
        t_stats_incorrect = []  # Incorrect > Correct direction
        
        n_features = correct_features.shape[1]
        
        for i in range(n_features):
            correct_acts = correct_features[:, i].cpu().numpy()
            incorrect_acts = incorrect_features[:, i].cpu().numpy()
            
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
                logger.warning(f"T-test failed for feature {i}: {e}")
                t_stats_correct.append(0.0)
                t_stats_incorrect.append(0.0)
        
        return {
            't_stats_correct': t_stats_correct,
            't_stats_incorrect': t_stats_incorrect
        }
    
    def analyze_layer(self, layer_idx: int) -> Dict:
        """Analyze a single layer for PVA directions using t-statistics."""
        logger.info(f"Analyzing layer {layer_idx}")
        
        # Load SAE for this layer
        sae = load_gemma_scope_sae(layer_idx, self.device)
        
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
            correct_features = sae.encode(correct_activations)
            incorrect_features = sae.encode(incorrect_activations)
        
        # DEBUG: Check SAE feature statistics
        logger.info(f"Layer {layer_idx} SAE features:")
        logger.info(f"  Correct features: mean={correct_features.mean():.6f}, std={correct_features.std():.6f}")
        logger.info(f"  Incorrect features: mean={incorrect_features.mean():.6f}, std={incorrect_features.std():.6f}")
        logger.info(f"  Active correct features: {(correct_features > 0).sum()}/{correct_features.numel()}")
        logger.info(f"  Active incorrect features: {(incorrect_features > 0).sum()}/{incorrect_features.numel()}")
        
        # Compute t-statistics
        t_stats = self.compute_t_statistics(correct_features, incorrect_features)
        
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
        
        # Store ALL features for global selection
        num_features = len(t_stats['t_stats_correct'])
        features_correct = []
        features_incorrect = []
        
        for i in range(num_features):
            # For correct-preferring features
            features_correct.append({
                'feature_idx': i,
                't_statistic': t_stats['t_stats_correct'][i]
            })
            
            # For incorrect-preferring features
            features_incorrect.append({
                'feature_idx': i,
                't_statistic': t_stats['t_stats_incorrect'][i]
            })
        
        # Prepare results
        results = {
            'layer': layer_idx,
            'n_correct': len(correct_activations),
            'n_incorrect': len(incorrect_activations),
            'features': {
                'correct': features_correct,
                'incorrect': features_incorrect
            }
        }
        
        # Log summary statistics
        max_correct_t = max(t_stats['t_stats_correct'])
        max_incorrect_t = max(t_stats['t_stats_incorrect'])
        logger.info(
            f"Layer {layer_idx}: Processed {num_features} features. "
            f"Max correct t-stat={max_correct_t:.3f}, "
            f"Max incorrect t-stat={max_incorrect_t:.3f}"
        )
        
        # Clean up to free memory
        del sae, correct_activations, incorrect_activations
        del correct_features, incorrect_features
        torch.cuda.empty_cache()
        
        return results
    
    def select_top_k_features_globally(self, all_results: Dict, k: int = 20) -> Dict:
        """Select top k features globally across all layers using t-statistics."""
        logger.info(f"Selecting top {k} features globally across all layers")
        
        # Collect all features from all layers
        all_features_correct = []
        all_features_incorrect = []
        
        for layer_idx, layer_results in all_results.items():
            for feature in layer_results['features']['correct']:
                feature_with_layer = feature.copy()
                feature_with_layer['layer'] = layer_idx
                all_features_correct.append(feature_with_layer)
                
            for feature in layer_results['features']['incorrect']:
                feature_with_layer = feature.copy()
                feature_with_layer['layer'] = layer_idx
                all_features_incorrect.append(feature_with_layer)
        
        # Sort globally by t-statistic (higher is better)
        # Use layer and feature_idx as secondary keys for deterministic ordering
        # Note: We don't bias toward any particular layer - just use natural ordering
        top_correct = sorted(
            all_features_correct, 
            key=lambda x: (-x['t_statistic'], x['layer'], x['feature_idx'])
        )[:k]
        
        top_incorrect = sorted(
            all_features_incorrect, 
            key=lambda x: (-x['t_statistic'], x['layer'], x['feature_idx'])
        )[:k]
        
        # Log distribution of top features across layers
        correct_layer_counts = {}
        incorrect_layer_counts = {}
        
        for feat in top_correct:
            layer = feat['layer']
            correct_layer_counts[layer] = correct_layer_counts.get(layer, 0) + 1
            
        for feat in top_incorrect:
            layer = feat['layer']
            incorrect_layer_counts[layer] = incorrect_layer_counts.get(layer, 0) + 1
            
        logger.info(f"Top {k} correct features by layer: {correct_layer_counts}")
        logger.info(f"Top {k} incorrect features by layer: {incorrect_layer_counts}")
        
        return {
            'correct': top_correct,
            'incorrect': top_incorrect,
            'layer_distribution': {
                'correct': correct_layer_counts,
                'incorrect': incorrect_layer_counts
            }
        }
    
    def load_pile_activations_for_layer(self, layer_idx: int) -> torch.Tensor:
        """Load pile activations for a specific layer from Phase 2.2."""
        pile_dir = Path(self.config.phase2_2_output_dir) / "pile_activations"
        
        if not pile_dir.exists():
            logger.warning(f"Pile activation directory not found at {pile_dir}")
            return None
            
        # Collect all pile activations for this layer
        activations = []
        pile_files = sorted(pile_dir.glob(f"*_layer_{layer_idx}.npz"))
        
        if not pile_files:
            logger.warning(f"No pile activations found for layer {layer_idx}")
            return None
            
        for file_path in pile_files:
            # Load activation
            data = np.load(file_path)
            activation = torch.from_numpy(data['activation'])
            activations.append(activation)
            
        if not activations:
            return None
            
        # Stack all activations
        return torch.stack(activations).to(self.device)
    
    def compute_pile_frequencies(self, pile_features: torch.Tensor) -> torch.Tensor:
        """Compute activation frequencies on pile dataset."""
        # Calculate fraction of pile samples where each feature activates
        freq_acts = (pile_features > 0).float().mean(dim=0)
        return freq_acts
    
    def apply_pile_filter(
        self, 
        top_features: Dict[str, List], 
        pile_frequencies: Dict[int, torch.Tensor]
    ) -> Dict[str, List]:
        """
        Apply pile filtering to remove general language features.
        
        Args:
            top_features: Dict with 'correct' and 'incorrect' lists of features
            pile_frequencies: Dict mapping layer_idx to frequency tensors
            
        Returns:
            Filtered features dict with same structure
        """
        if not self.config.pile_filter_enabled:
            logger.info("Pile filtering disabled, returning original features")
            return top_features
            
        logger.info(f"Applying pile filter with threshold {self.config.pile_threshold}")
        filtered = {'correct': [], 'incorrect': []}
        
        for category in ['correct', 'incorrect']:
            for feature in top_features[category]:
                layer = feature['layer']
                feat_idx = feature['feature_idx']
                
                # Check pile frequency if available
                if layer in pile_frequencies and pile_frequencies[layer] is not None:
                    pile_freq = pile_frequencies[layer][feat_idx].item()
                    
                    # Keep feature if it's below threshold (specific to code, not general)
                    if pile_freq < self.config.pile_threshold:
                        filtered[category].append(feature)
                    else:
                        logger.debug(
                            f"Filtered out {category} feature {feat_idx} from layer {layer}: "
                            f"pile frequency {pile_freq:.3f} >= {self.config.pile_threshold}"
                        )
                else:
                    # If no pile data available, keep the feature
                    filtered[category].append(feature)
                    
                # Stop if we have enough features
                if len(filtered[category]) >= 20:
                    break
                    
        logger.info(
            f"Pile filtering complete: {len(filtered['correct'])} correct, "
            f"{len(filtered['incorrect'])} incorrect features retained"
        )
        return filtered
    
    def run(self) -> Dict:
        """Run t-statistic based analysis on all specified layers."""
        logger.info("Starting Phase 2.10: T-Statistic Based Latent Selection")
        
        all_results = {}
        layer_summaries = []
        pile_frequencies = {}
        
        # Analyze each layer
        for layer_idx in tqdm(self.config.activation_layers, desc="Analyzing layers"):
            try:
                layer_results = self.analyze_layer(layer_idx)
                all_results[layer_idx] = layer_results
                layer_summaries.append(layer_results)
                
                # If pile filtering is enabled, compute pile frequencies
                if self.config.pile_filter_enabled:
                    logger.info(f"Loading pile activations for layer {layer_idx}")
                    pile_activations = self.load_pile_activations_for_layer(layer_idx)
                    
                    if pile_activations is not None:
                        # Load SAE for this layer
                        sae = load_gemma_scope_sae(layer_idx, self.device)
                        
                        # Ensure dtype matches SAE parameters for matrix multiplication
                        pile_activations = pile_activations.to(sae.W_enc.dtype)
                        
                        # Encode pile activations through SAE
                        with torch.no_grad():
                            pile_features = sae.encode(pile_activations)
                        
                        # Compute frequencies
                        pile_frequencies[layer_idx] = self.compute_pile_frequencies(pile_features)
                        
                        # Clean up
                        del sae, pile_activations, pile_features
                        torch.cuda.empty_cache()
                    else:
                        pile_frequencies[layer_idx] = None
                        
            except Exception as e:
                logger.error(f"Failed to analyze layer {layer_idx}: {e}")
                continue
        
        # Select top features globally (before filtering)
        top_features_unfiltered = self.select_top_k_features_globally(all_results, k=100)  # Get more to filter
        
        # Apply pile filtering if enabled
        if self.config.pile_filter_enabled and pile_frequencies:
            top_features = self.apply_pile_filter(top_features_unfiltered, pile_frequencies)
        else:
            # If no pile filtering, just take top 20
            top_features = {
                'correct': top_features_unfiltered['correct'][:20],
                'incorrect': top_features_unfiltered['incorrect'][:20],
                'layer_distribution': top_features_unfiltered.get('layer_distribution', {})
            }
        
        # Determine best layer (the one with most features in top 20)
        layer_counts = {}
        for feat in top_features['correct'] + top_features['incorrect']:
            layer = feat['layer']
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        best_layer = max(layer_counts.items(), key=lambda x: x[1])[0] if layer_counts else self.config.activation_layers[0]
        
        # Find best feature indices for the best layer
        best_correct_feature = next((f for f in top_features['correct'] if f['layer'] == best_layer), top_features['correct'][0])
        best_incorrect_feature = next((f for f in top_features['incorrect'] if f['layer'] == best_layer), top_features['incorrect'][0])
        
        # Prepare final results
        results = {
            'creation_timestamp': datetime.now().isoformat(),
            'model_name': self.config.model_name,
            'activation_layers': self.config.activation_layers,
            'layer_results': all_results,
            'top_20_features': top_features,
            'pile_filter_enabled': self.config.pile_filter_enabled,
            'pile_threshold': self.config.pile_threshold if self.config.pile_filter_enabled else None,
            'selection_method': 't_statistic',  # Mark this as t-statistic selection
            'best_layer': {
                'correct': best_layer,
                'incorrect': best_layer,
                'correct_feature_idx': best_correct_feature['feature_idx'],
                'incorrect_feature_idx': best_incorrect_feature['feature_idx']
            }
        }
        
        # Save results
        self._save_results(results)
        
        logger.info("Phase 2.10 completed. Top features selected using t-statistics.")
        return results
    
    def _save_results(self, results: Dict) -> None:
        """Save analysis results to file."""
        output_dir = Path(getattr(self.config, 'phase2_10_output_dir', 'data/phase2_10'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save per-layer features (complete rankings)
        for layer_idx, layer_data in results['layer_results'].items():
            layer_file = output_dir / f"layer_{layer_idx}_features.json"
            with open(layer_file, 'w') as f:
                json.dump({
                    'layer': layer_idx,
                    'n_correct': layer_data['n_correct'],
                    'n_incorrect': layer_data['n_incorrect'],
                    'features': layer_data['features']
                }, f, indent=2)
            logger.info(f"Saved layer {layer_idx} features to {layer_file}")
        
        # Save top 20 features
        top_features_file = output_dir / "top_20_features.json"
        with open(top_features_file, 'w') as f:
            json.dump(results['top_20_features'], f, indent=2)
        logger.info(f"Saved top 20 features to {top_features_file}")
        
        # Save best layer info (for Phase 3.5 compatibility)
        best_layer_file = output_dir / "best_layer.json"
        with open(best_layer_file, 'w') as f:
            json.dump(results['best_layer'], f, indent=2)
        logger.info(f"Saved best layer info to {best_layer_file}")
        
        # Save summary results (without layer_results to avoid huge file)
        summary_results = {
            'creation_timestamp': results['creation_timestamp'],
            'model_name': results['model_name'],
            'activation_layers': results['activation_layers'],
            'top_20_features': results['top_20_features'],
            'selection_method': results['selection_method'],
            'best_layer': results['best_layer']
        }
        
        output_file = output_dir / "sae_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        logger.info(f"Saved summary results to {output_file}")