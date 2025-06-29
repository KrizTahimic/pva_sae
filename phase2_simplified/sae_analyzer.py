"""
Simplified SAE analyzer for Phase 2.

Loads saved activations from Phase 1 and analyzes them using GemmaScope SAEs
to identify PVA latent directions.
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
logger = get_logger("sae_analyzer", phase="2")


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
        
        return {
            'f_correct': f_correct,
            'f_incorrect': f_incorrect,
            's_correct': s_correct,
            's_incorrect': s_incorrect
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
        
        # Find best features
        best_correct_idx = scores['s_correct'].argmax().item()
        best_incorrect_idx = scores['s_incorrect'].argmax().item()
        
        # Prepare results
        results = {
            'layer': layer_idx,
            'n_correct': len(correct_activations),
            'n_incorrect': len(incorrect_activations),
            'correct_direction': {
                'feature_idx': best_correct_idx,
                'separation_score': scores['s_correct'][best_correct_idx].item(),
                'f_correct': scores['f_correct'][best_correct_idx].item(),
                'f_incorrect': scores['f_incorrect'][best_correct_idx].item()
            },
            'incorrect_direction': {
                'feature_idx': best_incorrect_idx,
                'separation_score': scores['s_incorrect'][best_incorrect_idx].item(),
                'f_correct': scores['f_correct'][best_incorrect_idx].item(),
                'f_incorrect': scores['f_incorrect'][best_incorrect_idx].item()
            }
        }
        
        logger.info(
            f"Layer {layer_idx}: Best correct feature={best_correct_idx} "
            f"(score={scores['s_correct'][best_correct_idx]:.3f}), "
            f"Best incorrect feature={best_incorrect_idx} "
            f"(score={scores['s_incorrect'][best_incorrect_idx]:.3f})"
        )
        
        # Clean up to free memory
        del sae, correct_activations, incorrect_activations
        del correct_features, incorrect_features
        torch.cuda.empty_cache()
        
        return results
    
    def run(self) -> Dict:
        """Run SAE analysis on all specified layers."""
        logger.info("Starting Phase 2: SAE Analysis")
        
        all_results = {}
        layer_summaries = []
        
        # Analyze each layer
        for layer_idx in tqdm(self.config.activation_layers, desc="Analyzing layers"):
            try:
                layer_results = self.analyze_layer(layer_idx)
                all_results[layer_idx] = layer_results
                layer_summaries.append(layer_results)
            except Exception as e:
                logger.error(f"Failed to analyze layer {layer_idx}: {e}")
                continue
        
        # Find best layer overall
        best_layer = self._find_best_layer(all_results)
        
        # Prepare final results
        results = {
            'creation_timestamp': datetime.now().isoformat(),
            'model_name': self.config.model_name,
            'activation_layers': self.config.activation_layers,
            'layer_results': all_results,
            'best_layer_for_pva': best_layer,
            'summary': self._create_summary(all_results, best_layer)
        }
        
        # Save results
        self._save_results(results)
        
        logger.info(f"Phase 2 completed. Best PVA layer: {best_layer}")
        return results
    
    def _find_best_layer(self, all_results: Dict) -> int:
        """Find the layer with the best PVA separation."""
        best_score = -float('inf')
        best_layer = None
        
        for layer_idx, results in all_results.items():
            # Use the maximum of correct and incorrect separation scores
            max_score = max(
                results['correct_direction']['separation_score'],
                results['incorrect_direction']['separation_score']
            )
            
            if max_score > best_score:
                best_score = max_score
                best_layer = layer_idx
        
        return best_layer
    
    def _create_summary(self, all_results: Dict, best_layer: int) -> str:
        """Create a human-readable summary."""
        lines = []
        lines.append("SAE Analysis Summary")
        lines.append("=" * 50)
        
        for layer_idx in sorted(all_results.keys()):
            results = all_results[layer_idx]
            is_best = " (BEST)" if layer_idx == best_layer else ""
            
            lines.append(f"\nLayer {layer_idx}{is_best}:")
            lines.append(
                f"  Correct direction: feature {results['correct_direction']['feature_idx']} "
                f"(score: {results['correct_direction']['separation_score']:.3f})"
            )
            lines.append(
                f"  Incorrect direction: feature {results['incorrect_direction']['feature_idx']} "
                f"(score: {results['incorrect_direction']['separation_score']:.3f})"
            )
        
        return "\n".join(lines)
    
    def _save_results(self, results: Dict) -> None:
        """Save analysis results to file."""
        output_dir = Path(self.config.phase2_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        output_file = output_dir / "sae_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {output_file}")
        
        # Save best layer info for Phase 3.5
        best_layer_file = output_dir / "best_layer.json"
        with open(best_layer_file, 'w') as f:
            json.dump({'best_layer': results['best_layer_for_pva']}, f)
        
        logger.info(f"Saved best layer info to {best_layer_file}")