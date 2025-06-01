"""
SAE (Sparse Autoencoder) Analysis module for Phase 2 of the PVA-SAE project.

This module implements the core SAE analysis functionality:
1. Hook-based activation extraction
2. Separation score computation
3. PVA latent direction identification
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import logging
from huggingface_hub import hf_hub_download

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


def load_gemma_scope_sae(repo_id: str, sae_id: str, device: str = "cuda") -> JumpReLUSAE:
    """Load a GemmaScope SAE from HuggingFace Hub."""
    logger.info(f"Loading SAE from {repo_id}/{sae_id}")
    
    # Download parameters
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=f"{sae_id}/params.npz",
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


class ActivationExtractor:
    """Handles hook-based activation extraction from transformer models."""
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.activations_cache = {}
        
    def _create_hook(self, layer_name: str):
        """Create a hook function for capturing activations."""
        def hook_fn(module, input):
            # For pre-hooks, we only get module and input (no output)
            # For transformer blocks, we want the residual stream
            # This is typically the first element of the input tuple
            if isinstance(input, tuple):
                activation = input[0].detach()
            else:
                activation = input.detach()
            
            # Store only the last token's activation
            # Shape: [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
            last_token_acts = activation[:, -1, :].clone()
            
            if layer_name in self.activations_cache:
                self.activations_cache[layer_name].append(last_token_acts)
            else:
                self.activations_cache[layer_name] = [last_token_acts]
                
        return hook_fn
    
    def extract_activations(self, prompts: List[str], layer_idx: int) -> torch.Tensor:
        """Extract activations for given prompts at specified layer."""
        self.activations_cache.clear()
        
        # Get the layer module
        layer_name = f"layer_{layer_idx}"
        layer_module = self.model.model.layers[layer_idx]
        
        # Register hook
        hook_handle = layer_module.register_forward_pre_hook(
            self._create_hook(layer_name)
        )
        
        try:
            # Process prompts
            with torch.no_grad():
                for prompt in prompts:
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)
                    
                    # Forward pass (just to trigger hooks)
                    _ = self.model(**inputs)
            
            # Collect activations
            activations = torch.cat(self.activations_cache[layer_name], dim=0)
            logger.info(f"Extracted activations shape: {activations.shape}")
            
            return activations
            
        finally:
            # Always remove hook
            hook_handle.remove()


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
    model,
    tokenizer,
    sae: JumpReLUSAE,
    correct_prompts: List[str],
    incorrect_prompts: List[str],
    layer_idx: int,
    device: str = "cuda"
) -> Tuple[PVALatentDirection, PVALatentDirection]:
    """
    Find PVA latent directions for a single layer.
    
    Returns:
        Tuple of (correct_direction, incorrect_direction)
    """
    logger.info(f"Analyzing layer {layer_idx}")
    
    # Extract activations
    extractor = ActivationExtractor(model, tokenizer, device)
    
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