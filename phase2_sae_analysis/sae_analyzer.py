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
from typing import List, Tuple, Dict, Union, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import contextlib
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

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


def load_gemma_scope_sae(repo_id: str, sae_id: str, device: str = "mps") -> JumpReLUSAE:
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


class ActivationCache:
    """Thread-safe activation cache with automatic cleanup."""
    
    def __init__(self):
        self.cache = {}
    
    def store(self, key: str, activation: torch.Tensor):
        """Store activation with given key."""
        self.cache[key] = activation.detach().clone()
    
    def get(self, key: str) -> torch.Tensor:
        """Get activation by key."""
        return self.cache.get(key)
    
    def clear(self):
        """Clear all cached activations."""
        self.cache.clear()
    
    def keys(self):
        """Get cache keys."""
        return self.cache.keys()
    
    def __getitem__(self, key: str):
        return self.cache[key]
    
    def __contains__(self, key: str):
        return key in self.cache


class ActivationExtractor:
    """Activation extractor with robust error handling and multi-model support."""
    
    def __init__(
        self, 
        model: Union[HookedTransformer, torch.nn.Module], 
        tokenizer: AutoTokenizer,
        device: str = "mps"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.is_hooked_transformer = isinstance(model, HookedTransformer)
        self.cache = ActivationCache()
        
        logger.info(f"Initialized extractor for {'TransformerLens' if self.is_hooked_transformer else 'Hugging Face'} model")
    
    @contextlib.contextmanager
    def _hook_context(self, hooks: List[Tuple]):
        """Context manager for safe hook management."""
        handles = []
        try:
            if self.is_hooked_transformer:
                # TransformerLens hooks
                for hook_name, hook_fn in hooks:
                    handle = self.model.add_hook(hook_name, hook_fn)
                    handles.append(handle)
            else:
                # Hugging Face hooks
                for module, hook_fn in hooks:
                    handle = module.register_forward_pre_hook(hook_fn)
                    handles.append(handle)
            yield
        finally:
            # Clean up hooks
            for handle in handles:
                if self.is_hooked_transformer:
                    handle.remove()
                else:
                    handle.remove()
    
    def extract_activations(
        self,
        prompts: List[str],
        layer_idx: int,
        position: Union[int, str] = -1,
        hook_type: str = "resid_pre"
    ) -> torch.Tensor:
        """
        Extract activations for given prompts at specified layer and position.
        
        Args:
            prompts: List of text prompts
            layer_idx: Layer index to extract from
            position: Token position (-1 for last, 'all' for all tokens)
            hook_type: Type of hook for TransformerLens (resid_pre, resid_mid, resid_post)
            
        Returns:
            Tensor of extracted activations
        """
        if not prompts:
            raise ValueError("No prompts provided")
        
        self.cache.clear()
        
        if self.is_hooked_transformer:
            return self._extract_with_transformerlens(prompts, layer_idx, position, hook_type)
        else:
            return self._extract_with_huggingface(prompts, layer_idx, position)
    
    def _extract_with_transformerlens(
        self,
        prompts: List[str],
        layer_idx: int,
        position: Union[int, str],
        hook_type: str
    ) -> torch.Tensor:
        """Extract activations using TransformerLens hooks."""
        hook_name = f"blocks.{layer_idx}.hook_{hook_type}"
        
        def extraction_hook(activation, hook):
            """Hook function for TransformerLens."""
            if position == 'all':
                extracted = activation.detach().clone()
            elif isinstance(position, int):
                pos = position if position != -1 else activation.shape[1] - 1
                extracted = activation[:, pos, :].detach().clone()
            else:
                raise ValueError(f"Invalid position: {position}")
                
            # Store each sample separately for easier concatenation
            for i in range(extracted.shape[0]):
                cache_key = f"sample_{len(self.cache.cache)}"
                self.cache.store(cache_key, extracted[i:i+1])
            
            return activation
        
        # Use context manager for safe hook management
        with self._hook_context([(hook_name, extraction_hook)]):
            self._process_prompts_batch(prompts)
        
        return self._collect_cached_activations()
    
    def _extract_with_huggingface(
        self,
        prompts: List[str],
        layer_idx: int,
        position: Union[int, str]
    ) -> torch.Tensor:
        """Extract activations using Hugging Face hooks."""
        try:
            layer_module = self.model.model.layers[layer_idx]
        except (AttributeError, IndexError) as e:
            raise ValueError(f"Could not access layer {layer_idx}: {e}")
        
        def extraction_hook(module, input):
            """Hook function for Hugging Face models."""
            activation = input[0] if isinstance(input, tuple) else input
            
            if position == 'all':
                extracted = activation.detach().clone()
            elif isinstance(position, int):
                pos = position if position != -1 else activation.shape[1] - 1
                extracted = activation[:, pos, :].detach().clone()
            else:
                raise ValueError(f"Invalid position: {position}")
                
            # Store each sample separately
            for i in range(extracted.shape[0]):
                cache_key = f"sample_{len(self.cache.cache)}"
                self.cache.store(cache_key, extracted[i:i+1])
        
        with self._hook_context([(layer_module, extraction_hook)]):
            self._process_prompts_batch(prompts)
        
        return self._collect_cached_activations()
    
    def _process_prompts_batch(self, prompts: List[str], batch_size: int = 1):
        """Process prompts in batches to trigger hooks."""
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Forward pass to trigger hooks
                try:
                    if self.is_hooked_transformer:
                        _ = self.model(inputs.input_ids)
                    else:
                        _ = self.model(**inputs)
                except Exception as e:
                    logger.error(f"Error during forward pass: {e}")
                    raise
    
    def _collect_cached_activations(self) -> torch.Tensor:
        """Collect and concatenate all cached activations."""
        activations = []
        for key in sorted(self.cache.keys()):
            activations.append(self.cache[key])
        
        if not activations:
            raise RuntimeError("No activations were captured")
            
        result = torch.cat(activations, dim=0)
        logger.info(f"Extracted activations shape: {result.shape}")
        return result


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
    device: str = "mps"
) -> Tuple[PVALatentDirection, PVALatentDirection]:
    """
    Find PVA latent directions for a single layer.
    
    Returns:
        Tuple of (correct_direction, incorrect_direction)
    """
    logger.info(f"Analyzing layer {layer_idx}")
    
    # Validate inputs
    if not correct_prompts or not incorrect_prompts:
        raise ValueError("Both correct and incorrect prompts must be provided")
    
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


@dataclass
class SAEAnalysisResults:
    """Container for complete SAE analysis results."""
    layer_idx: int
    correct_direction: PVALatentDirection
    incorrect_direction: PVALatentDirection
    separation_scores: SeparationScores
    correct_sae_activations: torch.Tensor
    incorrect_sae_activations: torch.Tensor
    
    def summary(self) -> str:
        """Get a summary of the analysis results."""
        return (
            f"SAE Analysis Results for Layer {self.layer_idx}:\n"
            f"  {self.correct_direction}\n"
            f"  {self.incorrect_direction}\n"
            f"  Correct samples: {self.correct_sae_activations.shape[0]}\n"
            f"  Incorrect samples: {self.incorrect_sae_activations.shape[0]}"
        )


class SAEAnalysisPipeline:
    """
    Integrated pipeline for complete SAE analysis workflow.
    
    This class orchestrates the entire SAE analysis process:
    1. Model and SAE loading
    2. Activation extraction
    3. Separation score computation
    4. PVA direction identification
    """
    
    def __init__(
        self,
        model: Union[HookedTransformer, torch.nn.Module],
        tokenizer: AutoTokenizer,
        device: str = "mps"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.extractor = ActivationExtractor(model, tokenizer, device)
        
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
        correct_acts = self.extractor.extract_activations(correct_prompts, layer_idx)
        
        logger.info(f"Extracting activations for {len(incorrect_prompts)} incorrect samples")  
        incorrect_acts = self.extractor.extract_activations(incorrect_prompts, layer_idx)
        
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