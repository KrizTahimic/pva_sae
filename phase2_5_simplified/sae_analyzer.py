"""
Simplified SAE analyzer for Phase 2.5.

Loads saved activations from Phase 1 and analyzes them using GemmaScope SAEs
to identify PVA latent directions. Applies pile filtering to remove general
language features.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from huggingface_hub import hf_hub_download

from common.config import Config, GEMMA_2B_SPARSITY
from common.logging import get_logger

# Module-level logger
logger = get_logger("sae_analyzer", phase="2.5")


class JumpReLUSAE(torch.nn.Module):
    """JumpReLU Sparse Autoencoder implementation."""
    
    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.W_enc = torch.nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = torch.nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = x @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts


def load_gemma_scope_sae(layer_idx: int, device: str) -> JumpReLUSAE:
    """Load a GemmaScope SAE for a specific layer."""
    logger.info(f"Loading GemmaScope SAE for layer {layer_idx}")
    
    # GemmaScope repository
    repo_id = "google/gemma-scope-2b-pt-res"
    
    # Get the correct sparsity level for this layer
    if layer_idx not in GEMMA_2B_SPARSITY:
        raise ValueError(f"No sparsity mapping for layer {layer_idx}")
    
    sparsity = GEMMA_2B_SPARSITY[layer_idx]
    
    # Path within repository for this layer
    sae_path = f"layer_{layer_idx}/width_16k/average_l0_{sparsity}/params.npz"
    
    logger.info(f"Loading from path: {sae_path}")
    
    # Download parameters
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=sae_path,
        force_download=False,
    )
    
    # Load parameters
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}
    
    # Create and initialize SAE
    d_model = params['W_enc'].shape[0]
    d_sae = params['W_enc'].shape[1]
    sae = JumpReLUSAE(d_model, d_sae)
    sae.load_state_dict(pt_params)
    sae.to(device)
    
    logger.info(f"Loaded SAE with d_model={d_model}, d_sae={d_sae}, sparsity={sparsity}")
    return sae


class SimplifiedSAEAnalyzer:
    """Simplified SAE analyzer without complex abstractions."""
    
    def __init__(self, config: Config):
        """Initialize analyzer with configuration."""
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
    
    def compute_separation_scores(
        self,
        correct_features: torch.Tensor,
        incorrect_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute separation scores for PVA identification."""
        # Calculate activation fractions
        f_correct = (correct_features > 0).float().mean(dim=0)
        f_incorrect = (incorrect_features > 0).float().mean(dim=0)
        
        # Calculate separation scores
        s_correct = f_correct - f_incorrect
        s_incorrect = f_incorrect - f_correct
        
        # Calculate mean activations
        mean_correct = correct_features.mean(dim=0)
        mean_incorrect = incorrect_features.mean(dim=0)
        
        return {
            'f_correct': f_correct,
            'f_incorrect': f_incorrect,
            's_correct': s_correct,
            's_incorrect': s_incorrect,
            'mean_correct': mean_correct,
            'mean_incorrect': mean_incorrect
        }
    
    def analyze_layer(self, layer_idx: int) -> Dict:
        """Analyze a single layer for PVA directions."""
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
        
        # Encode activations through SAE
        with torch.no_grad():
            correct_features = sae.encode(correct_activations)
            incorrect_features = sae.encode(incorrect_activations)
        
        # Compute separation scores
        scores = self.compute_separation_scores(correct_features, incorrect_features)
        
        # Store ALL features for global selection
        num_features = scores['s_correct'].shape[0]
        features_correct = []
        features_incorrect = []
        
        for i in range(num_features):
            features_correct.append({
                'feature_idx': i,
                'separation_score': scores['s_correct'][i].item(),
                'f_correct': scores['f_correct'][i].item(),
                'f_incorrect': scores['f_incorrect'][i].item(),
                'mean_activation': scores['mean_correct'][i].item()
            })
            features_incorrect.append({
                'feature_idx': i,
                'separation_score': scores['s_incorrect'][i].item(),
                'f_correct': scores['f_correct'][i].item(),
                'f_incorrect': scores['f_incorrect'][i].item(),
                'mean_activation': scores['mean_incorrect'][i].item()
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
        max_correct_score = scores['s_correct'].max().item()
        max_incorrect_score = scores['s_incorrect'].max().item()
        logger.info(
            f"Layer {layer_idx}: Processed {num_features} features. "
            f"Max correct score={max_correct_score:.3f}, "
            f"Max incorrect score={max_incorrect_score:.3f}"
        )
        
        # Clean up to free memory
        del sae, correct_activations, incorrect_activations
        del correct_features, incorrect_features
        torch.cuda.empty_cache()
        
        return results
    
    def select_top_k_features_globally(self, all_results: Dict, k: int = 20) -> Dict:
        """Select top k features globally across all layers."""
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
        
        # Sort globally by separation score and take top k
        top_correct = sorted(
            all_features_correct, 
            key=lambda x: x['separation_score'], 
            reverse=True
        )[:k]
        
        top_incorrect = sorted(
            all_features_incorrect, 
            key=lambda x: x['separation_score'], 
            reverse=True
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
        """Run SAE analysis on all specified layers."""
        logger.info("Starting Phase 2.5: SAE Analysis with Pile Filtering")
        
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
        
        # Prepare final results
        results = {
            'creation_timestamp': datetime.now().isoformat(),
            'model_name': self.config.model_name,
            'activation_layers': self.config.activation_layers,
            'layer_results': all_results,
            'top_20_features': top_features,
            'pile_filter_enabled': self.config.pile_filter_enabled,
            'pile_threshold': self.config.pile_threshold if self.config.pile_filter_enabled else None
        }
        
        # Save results
        self._save_results(results)
        
        logger.info("Phase 2.5 completed. Top features selected with pile filtering.")
        return results
    
    def _save_results(self, results: Dict) -> None:
        """Save analysis results to file."""
        output_dir = Path(self.config.phase2_5_output_dir)
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
        
        # Save summary results (without layer_results to avoid huge file)
        summary_results = {
            'creation_timestamp': results['creation_timestamp'],
            'model_name': results['model_name'],
            'activation_layers': results['activation_layers'],
            'top_20_features': results['top_20_features']
        }
        
        output_file = output_dir / "sae_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        logger.info(f"Saved summary results to {output_file}")