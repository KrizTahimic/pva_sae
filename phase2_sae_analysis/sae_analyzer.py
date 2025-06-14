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
from common.config import Config
from common.utils import memory_mapped_array, torch_memory_cleanup, torch_no_grad_and_cleanup

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


# Activation extraction is handled in Phase 1 during generation
# Phase 2 only loads saved activations from disk


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


# Removed find_pva_directions - not used in Phase 2 (YAGNI)


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


# Removed SAEAnalysisPipeline - not used in Phase 2 (YAGNI)
# Phase 2 loads saved activations from disk, doesn't need model or extraction pipeline


# Removed unused pipeline classes (YAGNI) - Phase 2 only needs to load saved activations
# and run them through SAEs. No model loading or activation extraction needed.

class MultiLayerSAEResults:
    """Container for multi-layer SAE analysis results"""
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


# Removed MultiLayerSAEAnalyzer and EnhancedSAEAnalysisPipeline (YAGNI)
# Phase 2 implementation in run.py directly orchestrates the analysis
