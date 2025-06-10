"""
SAE (Sparse Autoencoder) Analysis module for Phase 2 of the PVA-SAE project.

This module implements the complete SAE analysis pipeline:
1. Hook-based activation extraction with robust error handling
2. Separation score computation following thesis methodology
3. PVA latent direction identification
4. Integrated pipeline for end-to-end analysis
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import contextlib
import json
import gc
import os
from datetime import datetime
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer

from common.config import SAELayerConfig, ActivationExtractionConfig
from common.utils import memory_mapped_array, torch_memory_cleanup, torch_no_grad_and_cleanup
from common.activation_extraction import (
    TransformerLensExtractor,
    HuggingFaceExtractor,
    ActivationData,
    create_activation_extractor
)

logger = logging.getLogger(__name__)


@dataclass
class SeparationScores:
    """Container for separation score analysis results."""
    f_correct: torch.Tensor  # Fraction of correct samples with non-zero activation
    f_incorrect: torch.Tensor  # Fraction of incorrect samples with non-zero activation
    s_correct: torch.Tensor  # Separation score for correct (f_correct - f_incorrect)
    s_incorrect: torch.Tensor  # Separation score for incorrect (f_incorrect - f_correct)
    
    @property
    def best_correct_idx(self) -> int:
        """Index of feature with highest s_correct."""
        return self.s_correct.argmax().item()
    
    @property
    def best_incorrect_idx(self) -> int:
        """Index of feature with highest s_incorrect."""
        return self.s_incorrect.argmax().item()


@dataclass
class PVALatentDirection:
    """Represents a Program Validity Awareness latent direction."""
    direction_type: str  # "correct" or "incorrect"
    layer: int
    feature_idx: int
    separation_score: float
    f_correct: float
    f_incorrect: float
    
    def __str__(self):
        return (f"PVA {self.direction_type.capitalize()} Direction: "
                f"Layer {self.layer}, Feature {self.feature_idx}, "
                f"Score={self.separation_score:.3f}")


class JumpReLUSAE(nn.Module):
    """JumpReLU Sparse Autoencoder implementation."""
    
    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = x @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts
    
    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts = self.encode(x)
        recon = self.decode(acts)
        return recon


def load_gemma_scope_sae(repo_id: str, sae_id: str, device: str) -> JumpReLUSAE:
    """Load a GemmaScope SAE from HuggingFace Hub."""
    logger.info(f"Loading SAE from {repo_id}/{sae_id}")
    
    # GemmaScope SAEs have a specific directory structure
    # Format: layer_X/width_16k/average_l0_Y/params.npz
    if "/" not in sae_id:
        # If just layer name provided, use canonical structure
        sae_path = f"{sae_id}/width_16k/average_l0_71/params.npz"
    else:
        # Full path provided
        sae_path = f"{sae_id}/params.npz"
    
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
    
    logger.info(f"Loaded SAE with d_model={d_model}, d_sae={d_sae}")
    return sae


# ActivationCache has been moved to common.activation_extraction


# ActivationExtractor has been moved to common.activation_extraction
# We use the centralized TransformerLensExtractor instead
# Methods moved to common.activation_extraction


def compute_separation_scores(
    correct_activations: torch.Tensor,
    incorrect_activations: torch.Tensor
) -> SeparationScores:
    """
    Compute separation scores according to thesis methodology.
    
    Args:
        correct_activations: SAE activations for correct samples [n_correct, d_sae]
        incorrect_activations: SAE activations for incorrect samples [n_incorrect, d_sae]
    
    Returns:
        SeparationScores object containing f_correct, f_incorrect, s_correct, s_incorrect
    """
    # Calculate activation fractions
    f_correct = (correct_activations > 0).float().mean(dim=0)
    f_incorrect = (incorrect_activations > 0).float().mean(dim=0)
    
    # Calculate separation scores
    s_correct = f_correct - f_incorrect
    s_incorrect = f_incorrect - f_correct
    
    return SeparationScores(
        f_correct=f_correct,
        f_incorrect=f_incorrect,
        s_correct=s_correct,
        s_incorrect=s_incorrect
    )


def find_pva_directions(
    model: HookedTransformer,
    sae: JumpReLUSAE,
    correct_prompts: List[str],
    incorrect_prompts: List[str],
    layer_idx: int,
    device: str
) -> Tuple[PVALatentDirection, PVALatentDirection]:
    """
    Find PVA latent directions for a single layer.
    
    Args:
        model: TransformerLens HookedTransformer model
        sae: SAE model for the specified layer
        correct_prompts: List of prompts that produce correct solutions
        incorrect_prompts: List of prompts that produce incorrect solutions
        layer_idx: Layer index to analyze
        device: Device to run computations on
    
    Returns:
        Tuple of (correct_direction, incorrect_direction)
    """
    logger.info(f"Analyzing layer {layer_idx}")
    
    # Validate inputs
    if not correct_prompts or not incorrect_prompts:
        raise ValueError("Both correct and incorrect prompts must be provided")
    
    
    # Extract activations
    extractor = ActivationExtractor(model, device)
    
    logger.info(f"Extracting activations for {len(correct_prompts)} correct samples")
    correct_acts = extractor.extract_activations(correct_prompts, layer_idx)
    
    logger.info(f"Extracting activations for {len(incorrect_prompts)} incorrect samples")
    incorrect_acts = extractor.extract_activations(incorrect_prompts, layer_idx)
    
    # Apply SAE
    with torch.no_grad():
        correct_sae_acts = sae.encode(correct_acts)
        incorrect_sae_acts = sae.encode(incorrect_acts)
    
    # Compute separation scores
    scores = compute_separation_scores(correct_sae_acts, incorrect_sae_acts)
    
    # Create PVA directions
    correct_direction = PVALatentDirection(
        direction_type="correct",
        layer=layer_idx,
        feature_idx=scores.best_correct_idx,
        separation_score=scores.s_correct[scores.best_correct_idx].item(),
        f_correct=scores.f_correct[scores.best_correct_idx].item(),
        f_incorrect=scores.f_incorrect[scores.best_correct_idx].item()
    )
    
    incorrect_direction = PVALatentDirection(
        direction_type="incorrect",
        layer=layer_idx,
        feature_idx=scores.best_incorrect_idx,
        separation_score=scores.s_incorrect[scores.best_incorrect_idx].item(),
        f_correct=scores.f_correct[scores.best_incorrect_idx].item(),
        f_incorrect=scores.f_incorrect[scores.best_incorrect_idx].item()
    )
    
    return correct_direction, incorrect_direction


@dataclass
class SAEAnalysisResults:
    """Container for complete SAE analysis results."""
    layer_idx: int
    correct_direction: PVALatentDirection
    incorrect_direction: PVALatentDirection
    separation_scores: SeparationScores
    correct_sae_activations: torch.Tensor
    incorrect_sae_activations: torch.Tensor
    pile_filtered_correct: Optional[Dict[str, Any]] = None
    pile_filtered_incorrect: Optional[Dict[str, Any]] = None
    
    def summary(self) -> str:
        """Get a summary of the analysis results."""
        summary = (
            f"SAE Analysis Results for Layer {self.layer_idx}:\n"
            f"  {self.correct_direction}\n"
            f"  {self.incorrect_direction}\n"
            f"  Correct samples: {self.correct_sae_activations.shape[0]}\n"
            f"  Incorrect samples: {self.incorrect_sae_activations.shape[0]}"
        )
        
        # Add Pile filtering info if available
        if self.pile_filtered_correct:
            summary += f"\n  Pile-filtered correct: {self.pile_filtered_correct['n_filtered']} latents"
            if self.pile_filtered_correct['top_latent'] is not None:
                summary += f" (top: {self.pile_filtered_correct['top_latent']})"
        if self.pile_filtered_incorrect:
            summary += f"\n  Pile-filtered incorrect: {self.pile_filtered_incorrect['n_filtered']} latents"
            if self.pile_filtered_incorrect['top_latent'] is not None:
                summary += f" (top: {self.pile_filtered_incorrect['top_latent']})"
        
        return summary


class SAEAnalysisPipeline:
    """
    Integrated pipeline for complete SAE analysis workflow using TransformerLens.
    
    This class orchestrates the entire SAE analysis process:
    1. Model and SAE loading
    2. Activation extraction
    3. Separation score computation
    4. PVA direction identification
    """
    
    def __init__(
        self,
        model: HookedTransformer,
        device: str
    ):
        self.model = model
        self.device = device
        
        # Use centralized activation extractor
        activation_config = ActivationExtractionConfig(
            batch_size=8,
            max_cache_size_gb=10.0,
            clear_cache_between_layers=True
        )
        self.extractor = create_activation_extractor(
            model=model,
            model_type="transformerlens",
            device=device,
            config=activation_config
        )
        
        logger.info("Initialized SAE Analysis Pipeline")
    
    def load_sae(
        self,
        repo_id: str = "google/gemma-scope-2b-pt-res",
        layer_idx: int = 20,
        sae_id: Optional[str] = None
    ) -> JumpReLUSAE:
        """
        Load SAE for specified layer.
        
        Args:
            repo_id: HuggingFace repository ID
            layer_idx: Layer index for SAE
            sae_id: Specific SAE ID, defaults to layer-{layer_idx}
            
        Returns:
            Loaded SAE model
        """
        if sae_id is None:
            sae_id = f"layer_{layer_idx}"
        
        try:
            sae = load_gemma_scope_sae(repo_id, sae_id, self.device)
            logger.info(f"Successfully loaded SAE for layer {layer_idx}")
            return sae
        except Exception as e:
            logger.error(f"Failed to load SAE: {e}")
            raise
    
    def analyze_layer(
        self,
        correct_prompts: List[str],
        incorrect_prompts: List[str],
        layer_idx: int,
        sae: Optional[JumpReLUSAE] = None,
        repo_id: str = "google/gemma-scope-2b-pt-res"
    ) -> SAEAnalysisResults:
        """
        Perform complete SAE analysis for a single layer.
        
        Args:
            correct_prompts: List of prompts that produce correct solutions
            incorrect_prompts: List of prompts that produce incorrect solutions
            layer_idx: Layer to analyze
            sae: Pre-loaded SAE model (optional)
            repo_id: HuggingFace repository for SAE loading
            
        Returns:
            Complete analysis results
        """
        # Validate inputs
        if not correct_prompts or not incorrect_prompts:
            raise ValueError("Both correct and incorrect prompts must be provided")
        
        if len(correct_prompts) < 5 or len(incorrect_prompts) < 5:
            logger.warning("Small sample sizes may lead to unreliable results")
        
        # Load SAE if not provided
        if sae is None:
            sae = self.load_sae(repo_id, layer_idx)
        
        logger.info(f"Starting analysis for layer {layer_idx}")
        
        # Extract activations
        logger.info(f"Extracting activations for {len(correct_prompts)} correct samples")
        correct_data = self.extractor.extract_activations(
            prompts=correct_prompts,
            layer_idx=layer_idx,
            position=-1,  # Final token
            hook_type="resid_post"
        )
        correct_acts = correct_data.activations
        
        logger.info(f"Extracting activations for {len(incorrect_prompts)} incorrect samples")  
        incorrect_data = self.extractor.extract_activations(
            prompts=incorrect_prompts,
            layer_idx=layer_idx,
            position=-1,  # Final token
            hook_type="resid_post"
        )
        incorrect_acts = incorrect_data.activations
        
        # Apply SAE encoding
        with torch.no_grad():
            correct_sae_acts = sae.encode(correct_acts)
            incorrect_sae_acts = sae.encode(incorrect_acts)
        
        logger.info(f"SAE activations - Correct: {correct_sae_acts.shape}, Incorrect: {incorrect_sae_acts.shape}")
        
        # Compute separation scores
        scores = compute_separation_scores(correct_sae_acts, incorrect_sae_acts)
        
        # Create PVA directions
        correct_direction = PVALatentDirection(
            direction_type="correct",
            layer=layer_idx,
            feature_idx=scores.best_correct_idx,
            separation_score=scores.s_correct[scores.best_correct_idx].item(),
            f_correct=scores.f_correct[scores.best_correct_idx].item(),
            f_incorrect=scores.f_incorrect[scores.best_correct_idx].item()
        )
        
        incorrect_direction = PVALatentDirection(
            direction_type="incorrect",
            layer=layer_idx,
            feature_idx=scores.best_incorrect_idx,
            separation_score=scores.s_incorrect[scores.best_incorrect_idx].item(),
            f_correct=scores.f_correct[scores.best_incorrect_idx].item(),
            f_incorrect=scores.f_incorrect[scores.best_incorrect_idx].item()
        )
        
        results = SAEAnalysisResults(
            layer_idx=layer_idx,
            correct_direction=correct_direction,
            incorrect_direction=incorrect_direction,
            separation_scores=scores,
            correct_sae_activations=correct_sae_acts,
            incorrect_sae_activations=incorrect_sae_acts
        )
        
        logger.info(f"Analysis complete for layer {layer_idx}")
        logger.info(f"Best correct feature: {correct_direction.feature_idx} (score: {correct_direction.separation_score:.3f})")
        logger.info(f"Best incorrect feature: {incorrect_direction.feature_idx} (score: {incorrect_direction.separation_score:.3f})")
        
        return results
    
    def analyze_multiple_layers(
        self,
        correct_prompts: List[str],
        incorrect_prompts: List[str],
        layer_indices: List[int],
        repo_id: str = "google/gemma-scope-2b-pt-res"
    ) -> Dict[int, SAEAnalysisResults]:
        """
        Perform SAE analysis across multiple layers.
        
        Args:
            correct_prompts: List of prompts that produce correct solutions
            incorrect_prompts: List of prompts that produce incorrect solutions
            layer_indices: List of layer indices to analyze
            repo_id: HuggingFace repository for SAE loading
            
        Returns:
            Dictionary mapping layer_idx to analysis results
        """
        results = {}
        
        for layer_idx in layer_indices:
            try:
                results[layer_idx] = self.analyze_layer(
                    correct_prompts, incorrect_prompts, layer_idx, repo_id=repo_id
                )
            except Exception as e:
                logger.error(f"Failed to analyze layer {layer_idx}: {e}")
                continue
        
        logger.info(f"Completed analysis for {len(results)} layers")
        return results


class MultiLayerActivationCache:
    """Memory-efficient cache for multi-layer activations with optional memory mapping"""
    
    def __init__(self, device: str = "auto", use_memory_mapping: bool = False, 
                 cache_dir: Optional[str] = None, n_samples: int = None, 
                 n_layers: int = None, d_model: int = None):
        
        self.device = device
        self.use_memory_mapping = use_memory_mapping
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if use_memory_mapping:
            if not all([cache_dir, n_samples, n_layers, d_model]):
                raise ValueError("Memory mapping requires cache_dir, n_samples, n_layers, and d_model")
            self._setup_memory_mapping(n_samples, n_layers, d_model)
        else:
            self.activations = {}  # layer_idx -> activations tensor
    
    def _setup_memory_mapping(self, n_samples: int, n_layers: int, d_model: int):
        """Setup memory-mapped storage for large-scale analysis"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Store parameters for context manager
        self.mmap_params = {
            'filename': str(self.cache_dir / "activations.dat"),
            'dtype': np.float32,
            'shape': (n_samples, n_layers, d_model),
            'mode': 'w+'
        }
        
        # Keep track of which layers have data
        self.layer_sample_counts = {}
        
        logger.info(f"Prepared memory-mapped cache: {n_samples} samples x {n_layers} layers x {d_model} features")
    
    def store_layer(self, layer_idx: int, activations: torch.Tensor):
        """Store activations for a specific layer"""
        if self.use_memory_mapping:
            # Convert to numpy and store in memory-mapped array
            with memory_mapped_array(**self.mmap_params) as mmap:
                acts_np = activations.detach().cpu().numpy()
                n_samples = acts_np.shape[0]
                mmap[:n_samples, layer_idx, :] = acts_np
                self.layer_sample_counts[layer_idx] = n_samples
        else:
            # Store in regular memory
            self.activations[layer_idx] = activations.detach().clone()
    
    def get_layer(self, layer_idx: int) -> torch.Tensor:
        """Get activations for a specific layer"""
        if self.use_memory_mapping:
            if layer_idx not in self.layer_sample_counts:
                return None
            n_samples = self.layer_sample_counts[layer_idx]
            with memory_mapped_array(**self.mmap_params) as mmap:
                acts_np = mmap[:n_samples, layer_idx, :]
                return torch.from_numpy(acts_np.copy()).to(self.device)
        else:
            return self.activations.get(layer_idx)
    
    def clear_layer(self, layer_idx: int):
        """Clear activations for a specific layer to free memory"""
        if self.use_memory_mapping:
            if layer_idx in self.layer_sample_counts:
                del self.layer_sample_counts[layer_idx]
        else:
            if layer_idx in self.activations:
                del self.activations[layer_idx]
        
        # Use context manager for proper cleanup
        with torch_memory_cleanup(self.device):
            pass
    
    def cleanup(self):
        """Clean up all resources"""
        if self.use_memory_mapping:
            # Memory map cleanup is handled by context manager
            self.layer_sample_counts.clear()
        else:
            self.activations.clear()
        
        # Ensure GPU memory is cleaned
        with torch_memory_cleanup(self.device):
            pass


class MultiLayerActivationExtractor:
    """Extract activations from all specified layers in a single forward pass"""
    
    def __init__(self, model: HookedTransformer, device: str = "auto"):
        self.model = model
        self.device = device
        
        # Use centralized activation extractor
        activation_config = ActivationExtractionConfig(
            batch_size=1,
            max_cache_size_gb=20.0,
            clear_cache_between_layers=False  # Important for multi-layer extraction
        )
        self.extractor = create_activation_extractor(
            model=model,
            model_type="transformerlens",
            device=device,
            config=activation_config
        )
        
    @contextlib.contextmanager
    def _hook_context(self, hooks: List[Tuple]):
        """Context manager for safe hook management with TransformerLens"""
        try:
            # Clear any existing hooks first
            self.model.reset_hooks()
            
            # Add new hooks - TransformerLens manages them internally
            for hook_name, hook_fn in hooks:
                self.model.add_hook(hook_name, hook_fn)
            yield
        finally:
            # Clean up all hooks
            self.model.reset_hooks()
    
    def extract_all_layers(
        self,
        prompts: List[str],
        layer_indices: List[int],
        position: Union[int, str] = -1,
        hook_type: str = "resid_post",
        use_memory_mapping: bool = False,
        cache_dir: Optional[str] = None
    ) -> MultiLayerActivationCache:
        """Extract activations from all layers in one forward pass"""
        
        # Determine cache parameters
        if use_memory_mapping and cache_dir:
            # Get model dimensions for memory mapping
            d_model = self.model.cfg.d_model
            n_samples = len(prompts)
            n_layers = len(layer_indices)
            
            cache = MultiLayerActivationCache(
                device=self.device,
                use_memory_mapping=True,
                cache_dir=cache_dir,
                n_samples=n_samples,
                n_layers=n_layers,
                d_model=d_model
            )
        else:
            cache = MultiLayerActivationCache(device=self.device)
        
        # Create hooks for all layers
        hooks = []
        for layer_idx in layer_indices:
            hook_name = f"blocks.{layer_idx}.hook_{hook_type}"
            
            def make_hook(layer_idx):
                def extraction_hook(activation, hook):
                    if position == 'all':
                        extracted = activation.detach().clone()
                    elif isinstance(position, int):
                        pos = position if position != -1 else activation.shape[1] - 1
                        extracted = activation[:, pos, :].detach().clone()
                    else:
                        raise ValueError(f"Invalid position: {position}")
                    
                    cache.store_layer(layer_idx, extracted)
                    return activation
                return extraction_hook
            
            hooks.append((hook_name, make_hook(layer_idx)))
        
        # Single forward pass captures all layers
        with self._hook_context(hooks):
            self._process_prompts_batch(prompts)
        
        return cache
    
    def _process_prompts_batch(self, prompts: List[str]):
        """Process prompts in batches to trigger hooks"""
        # Use the extractor's batch processing (will trigger our hooks)
        self.extractor._process_prompts_batch(prompts)


@dataclass
class MultiLayerSAEResults:
    """Container for multi-layer SAE analysis results"""
    layer_results: Dict[int, SAEAnalysisResults]
    layer_indices: List[int]
    model_name: str
    n_correct_samples: int
    n_incorrect_samples: int
    hook_component: str
    
    def get_best_layer_for_correct(self) -> Tuple[int, PVALatentDirection]:
        """Find layer with best correct direction"""
        best_score = -float('inf')
        best_layer = None
        best_direction = None
        
        for layer_idx, results in self.layer_results.items():
            if results.correct_direction.separation_score > best_score:
                best_score = results.correct_direction.separation_score
                best_layer = layer_idx
                best_direction = results.correct_direction
        
        return best_layer, best_direction
    
    def get_best_layer_for_incorrect(self) -> Tuple[int, PVALatentDirection]:
        """Find layer with best incorrect direction"""
        best_score = -float('inf')
        best_layer = None
        best_direction = None
        
        for layer_idx, results in self.layer_results.items():
            if results.incorrect_direction.separation_score > best_score:
                best_score = results.incorrect_direction.separation_score
                best_layer = layer_idx
                best_direction = results.incorrect_direction
        
        return best_layer, best_direction
    
    def get_layer_summary(self) -> List[Dict]:
        """Get summary statistics for each layer"""
        summaries = []
        for layer_idx in sorted(self.layer_results.keys()):
            results = self.layer_results[layer_idx]
            summaries.append({
                'layer': layer_idx,
                'correct_feature': results.correct_direction.feature_idx,
                'correct_score': results.correct_direction.separation_score,
                'incorrect_feature': results.incorrect_direction.feature_idx,
                'incorrect_score': results.incorrect_direction.separation_score
            })
        return summaries
    
    def summary(self) -> str:
        """Get summary of all layer results"""
        best_correct_layer, best_correct = self.get_best_layer_for_correct()
        best_incorrect_layer, best_incorrect = self.get_best_layer_for_incorrect()
        
        return (
            f"Multi-Layer SAE Analysis Results:\n"
            f"  Model: {self.model_name}\n"
            f"  Component: {self.hook_component}\n"
            f"  Layers analyzed: {len(self.layer_results)} {sorted(self.layer_results.keys())}\n"
            f"  Samples: {self.n_correct_samples} correct, {self.n_incorrect_samples} incorrect\n"
            f"  Best correct direction: Layer {best_correct_layer}, "
            f"Feature {best_correct.feature_idx} (score: {best_correct.separation_score:.3f})\n"
            f"  Best incorrect direction: Layer {best_incorrect_layer}, "
            f"Feature {best_incorrect.feature_idx} (score: {best_incorrect.separation_score:.3f})"
        )
    
    def save_to_file(self, filepath: str):
        """Save results to JSON file"""
        data = {
            'model_name': self.model_name,
            'hook_component': self.hook_component,
            'n_correct_samples': self.n_correct_samples,
            'n_incorrect_samples': self.n_incorrect_samples,
            'layer_indices': self.layer_indices,
            'layer_summaries': self.get_layer_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved multi-layer results to {filepath}")


class MultiLayerSAEAnalyzer:
    """Analyze SAE features across multiple layers with memory management and checkpointing"""
    
    def __init__(self, model: HookedTransformer, sae_config: Optional[SAELayerConfig] = None, device: str = "auto"):
        self.model = model
        self.sae_config = sae_config or SAELayerConfig()
        
        
        self.device = device
        self.extractor = MultiLayerActivationExtractor(model, device)
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(self.sae_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized MultiLayerSAEAnalyzer for {model.cfg.model_name}")
    
    def analyze_all_layers(
        self,
        correct_prompts: List[str],
        incorrect_prompts: List[str],
        layer_indices: Optional[List[int]] = None
    ) -> MultiLayerSAEResults:
        """Analyze all specified layers with efficient memory management"""
        
        # Determine layers to analyze
        if layer_indices is None:
            n_layers = self.model.cfg.n_layers
            layer_indices = self.sae_config.get_layers_for_model(
                self.model.cfg.model_name, n_layers
            )
        
        logger.info(f"Analyzing {len(layer_indices)} layers: {layer_indices}")
        logger.info(f"Component: {self.sae_config.hook_component}")
        
        # Determine if memory mapping is needed
        use_memory_mapping = self.sae_config.should_use_memory_mapping(len(layer_indices))
        cache_dir = str(self.checkpoint_dir / "activation_cache") if use_memory_mapping else None
        
        # Step 1: Extract all activations in one forward pass
        logger.info("Extracting activations for correct samples...")
        correct_cache = self.extractor.extract_all_layers(
            correct_prompts, layer_indices, 
            hook_type=self.sae_config.hook_component,
            use_memory_mapping=use_memory_mapping,
            cache_dir=cache_dir + "_correct" if cache_dir else None
        )
        
        logger.info("Extracting activations for incorrect samples...")
        incorrect_cache = self.extractor.extract_all_layers(
            incorrect_prompts, layer_indices,
            hook_type=self.sae_config.hook_component,
            use_memory_mapping=use_memory_mapping,
            cache_dir=cache_dir + "_incorrect" if cache_dir else None
        )
        
        # Step 2: Process each layer sequentially
        results = {}
        for layer_idx in layer_indices:
            try:
                logger.info(f"Processing layer {layer_idx}...")
                
                # Get cached activations for this layer
                correct_acts = correct_cache.get_layer(layer_idx)
                incorrect_acts = incorrect_cache.get_layer(layer_idx)
                
                if correct_acts is None or incorrect_acts is None:
                    logger.error(f"No activations found for layer {layer_idx}")
                    continue
                
                # Load SAE for this layer
                sae = self._load_sae_for_layer(layer_idx)
                
                # Compute SAE activations
                with torch.no_grad():
                    correct_sae_acts = sae.encode(correct_acts)
                    incorrect_sae_acts = sae.encode(incorrect_acts)
                
                # Compute separation scores
                scores = compute_separation_scores(correct_sae_acts, incorrect_sae_acts)
                
                # Create results
                results[layer_idx] = self._create_analysis_results(
                    layer_idx, scores, correct_sae_acts, incorrect_sae_acts
                )
                
                # Save checkpoint after each layer if configured
                if self.sae_config.save_after_each_layer:
                    self._save_layer_checkpoint(layer_idx, results[layer_idx])
                
                # Critical: Clean up after each layer
                if self.sae_config.cleanup_after_layer:
                    del sae, correct_sae_acts, incorrect_sae_acts
                    correct_cache.clear_layer(layer_idx)
                    incorrect_cache.clear_layer(layer_idx)
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                
                logger.info(f"Layer {layer_idx} complete - best correct: "
                          f"{results[layer_idx].correct_direction.feature_idx} "
                          f"(score: {results[layer_idx].correct_direction.separation_score:.3f})")
                
            except Exception as e:
                logger.error(f"Failed to analyze layer {layer_idx}: {e}")
                continue
        
        # Clean up activation caches
        correct_cache.cleanup()
        incorrect_cache.cleanup()
        
        # Create comprehensive results
        multi_layer_results = MultiLayerSAEResults(
            layer_results=results,
            layer_indices=list(results.keys()),
            model_name=self.model.cfg.model_name,
            n_correct_samples=len(correct_prompts),
            n_incorrect_samples=len(incorrect_prompts),
            hook_component=self.sae_config.hook_component
        )
        
        # Save final results
        self._save_final_results(multi_layer_results)
        
        logger.info(f"Completed analysis for {len(results)} layers")
        return multi_layer_results
    
    def _load_sae_for_layer(self, layer_idx: int) -> JumpReLUSAE:
        """Load SAE for specific layer with dynamic ID generation"""
        sae_id = f"layer_{layer_idx}/width_{self.sae_config.sae_width}/average_l0_{self.sae_config.sae_sparsity}"
        
        return load_gemma_scope_sae(
            repo_id=self.sae_config.sae_repo_id,
            sae_id=sae_id,
            device=self.device
        )
    
    def _create_analysis_results(
        self,
        layer_idx: int,
        scores: SeparationScores,
        correct_sae_acts: torch.Tensor,
        incorrect_sae_acts: torch.Tensor
    ) -> SAEAnalysisResults:
        """Create analysis results for a single layer"""
        
        # Create PVA directions
        correct_direction = PVALatentDirection(
            direction_type="correct",
            layer=layer_idx,
            feature_idx=scores.best_correct_idx,
            separation_score=scores.s_correct[scores.best_correct_idx].item(),
            f_correct=scores.f_correct[scores.best_correct_idx].item(),
            f_incorrect=scores.f_incorrect[scores.best_correct_idx].item()
        )
        
        incorrect_direction = PVALatentDirection(
            direction_type="incorrect",
            layer=layer_idx,
            feature_idx=scores.best_incorrect_idx,
            separation_score=scores.s_incorrect[scores.best_incorrect_idx].item(),
            f_correct=scores.f_correct[scores.best_incorrect_idx].item(),
            f_incorrect=scores.f_incorrect[scores.best_incorrect_idx].item()
        )
        
        return SAEAnalysisResults(
            layer_idx=layer_idx,
            correct_direction=correct_direction,
            incorrect_direction=incorrect_direction,
            separation_scores=scores,
            correct_sae_activations=correct_sae_acts,
            incorrect_sae_activations=incorrect_sae_acts
        )
    
    def _save_layer_checkpoint(self, layer_idx: int, results: SAEAnalysisResults):
        """Save checkpoint after processing each layer"""
        checkpoint_file = self.checkpoint_dir / f"layer_{layer_idx}_results.json"
        
        data = {
            'layer_idx': layer_idx,
            'correct_direction': {
                'feature_idx': results.correct_direction.feature_idx,
                'separation_score': results.correct_direction.separation_score,
                'f_correct': results.correct_direction.f_correct,
                'f_incorrect': results.correct_direction.f_incorrect
            },
            'incorrect_direction': {
                'feature_idx': results.incorrect_direction.feature_idx,
                'separation_score': results.incorrect_direction.separation_score,
                'f_correct': results.incorrect_direction.f_correct,
                'f_incorrect': results.incorrect_direction.f_incorrect
            }
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved checkpoint for layer {layer_idx}")
    
    def _save_final_results(self, results: MultiLayerSAEResults):
        """Save final comprehensive results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.checkpoint_dir / f"multi_layer_results_{timestamp}.json"
        results.save_to_file(str(results_file))


class EnhancedSAEAnalysisPipeline(SAEAnalysisPipeline):
    """Enhanced pipeline with multi-layer support"""
    
    def __init__(self, model: HookedTransformer, sae_config: Optional[SAELayerConfig] = None, device: str = "auto"):
        super().__init__(model, device)
        self.sae_config = sae_config or SAELayerConfig()
        self.multi_layer_analyzer = MultiLayerSAEAnalyzer(model, self.sae_config, device)
    
    def analyze_all_residual_layers(
        self,
        correct_prompts: List[str],
        incorrect_prompts: List[str],
        layer_indices: Optional[List[int]] = None
    ) -> MultiLayerSAEResults:
        """Analyze resid_post across all specified layers"""
        
        logger.info(f"Starting comprehensive residual stream analysis...")
        
        return self.multi_layer_analyzer.analyze_all_layers(
            correct_prompts, incorrect_prompts, layer_indices
        )
    
    def apply_pile_filtering(
        self,
        results: MultiLayerSAEResults,
        pile_filter,
        pile_threshold: float = 0.02,
        top_k: int = 20
    ) -> MultiLayerSAEResults:
        """
        Apply Pile filtering to multi-layer results.
        
        Args:
            results: Multi-layer SAE analysis results
            pile_filter: PileFilter instance
            pile_threshold: Maximum allowed Pile activation frequency
            top_k: Maximum number of latents to keep after filtering
            
        Returns:
            Updated results with Pile filtering applied
        """
        from .pile_filter import PileFilter
        
        if not isinstance(pile_filter, PileFilter):
            raise ValueError("pile_filter must be a PileFilter instance")
        
        logger.info(f"Applying Pile filtering with threshold {pile_threshold}")
        
        # Process each layer
        for layer_idx in results.layer_results:
            layer_result = results.layer_results[layer_idx]
            
            logger.info(f"Filtering layer {layer_idx}...")
            
            try:
                # Load SAE for this layer
                sae = self.multi_layer_analyzer._load_sae_for_layer(layer_idx)
                
                # Compute Pile frequencies for this layer
                pile_frequencies = pile_filter.compute_pile_frequencies(layer_idx, sae)
                
                # Filter correct latents
                layer_result.pile_filtered_correct = pile_filter.filter_by_pile_frequency(
                    separation_scores=layer_result.separation_scores,
                    pile_frequencies=pile_frequencies,
                    direction='correct',
                    pile_threshold=pile_threshold,
                    top_k=top_k
                )
                
                # Filter incorrect latents
                layer_result.pile_filtered_incorrect = pile_filter.filter_by_pile_frequency(
                    separation_scores=layer_result.separation_scores,
                    pile_frequencies=pile_frequencies,
                    direction='incorrect',
                    pile_threshold=pile_threshold,
                    top_k=top_k
                )
                
                # Clean up
                del sae
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Failed to apply Pile filtering to layer {layer_idx}: {e}")
                continue
        
        logger.info("Pile filtering complete")
        return results