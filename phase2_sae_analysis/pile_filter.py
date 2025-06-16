"""
Pile dataset filtering for SAE latent analysis.

This module filters SAE latents based on their activation frequency on the Pile dataset,
helping identify entity-specific features vs general language patterns.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import json
import random
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm
import gc

from .sae_analyzer import SeparationScores
from common.utils import torch_memory_cleanup, torch_no_grad_and_cleanup, atomic_file_write
from common.utils import get_phase_dir
from common.logging import get_logger

# Module-level logger - uses global phase context
logger = get_logger("pile_filter")


class PileFilter:
    """Filters SAE latents based on Pile dataset activation frequencies."""
    
    def __init__(self, model, device: str = 'cuda'):
        """
        Initialize Pile filter.
        
        Args:
            model: HookedTransformer model for tokenization
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.pile_cache_dir = Path(get_phase_dir(2)) / "pile_cache"
        self.pile_cache_dir.mkdir(parents=True, exist_ok=True)
        self.eps = 1e-6  # Activation threshold for numerical stability
        
    def load_pile_dataset(self, n_samples: int = 10000) -> List[str]:
        """
        Load NeelNanda/pile-10k dataset from HuggingFace.
        
        Args:
            n_samples: Number of samples to load
            
        Returns:
            List of text samples
        """
        logger.info(f"Loading Pile dataset (up to {n_samples} samples)...")
        
        try:
            # Load the dataset
            dataset = load_dataset("NeelNanda/pile-10k", split='train')
            texts = dataset['text'][:n_samples]
            
            logger.info(f"Loaded {len(texts)} Pile samples")
            return texts
            
        except Exception as e:
            logger.error(f"Failed to load Pile dataset: {e}")
            raise
    
    def find_word_token_span(self, text: str, word: str, tokenized_text: torch.Tensor) -> Optional[Tuple[int, int]]:
        """
        Find token indices that span a word using binary search.
        
        Args:
            text: Full text
            word: Word to find
            tokenized_text: Tokenized version of text (1D tensor)
            
        Returns:
            Tuple of (start_idx, end_idx) or None if word not found
        """
        # Ensure we have a 1D tensor
        if tokenized_text.dim() > 1:
            tokenized_text = tokenized_text.squeeze()
        
        # Get token strings
        tokens = self.model.to_str_tokens(tokenized_text.unsqueeze(0) if tokenized_text.dim() == 1 else tokenized_text)
        
        # Binary search for word boundaries
        n_tokens = len(tokens)
        
        # Try each possible starting position
        for start_idx in range(n_tokens):
            # Try different end positions
            for end_idx in range(start_idx + 1, min(start_idx + 10, n_tokens + 1)):  # Limit search window
                # Reconstruct text from tokens
                reconstructed = ''.join(tokens[start_idx:end_idx])
                
                # Check if this matches our word
                if reconstructed.strip() == word:
                    return (start_idx, end_idx)
                    
                # Early termination if we've gone past the word
                if len(reconstructed.strip()) > len(word):
                    break
        
        return None
    
    def extract_word_activations(self, texts: List[str], layer_idx: int, sae_model) -> Dict[int, List[float]]:
        """
        Extract SAE activations for random words from texts.
        
        Args:
            texts: List of texts to process
            layer_idx: Layer index for activation extraction
            sae_model: SAE model for the specified layer
            
        Returns:
            Dictionary mapping latent indices to activation values
        """
        logger.info(f"Extracting activations for {len(texts)} texts at layer {layer_idx}")
        
        # Initialize activation storage
        latent_activations = {}  # latent_idx -> list of activations
        
        # Process texts in batches
        batch_size = 100
        valid_samples = 0
        
        for batch_start in tqdm(range(0, len(texts), batch_size), desc="Processing Pile texts"):
            batch_texts = texts[batch_start:batch_start + batch_size]
            
            for text in batch_texts:
                # Skip empty texts
                if not text or not text.strip():
                    continue
                
                # Split into words and select one randomly
                words = text.split()
                if not words:
                    continue
                    
                selected_word = random.choice(words)
                
                # Tokenize the text
                tokens = self.model.to_tokens(text, prepend_bos=True).to(self.device)
                
                # Find token span for the word
                span = self.find_word_token_span(text, selected_word, tokens[0])
                if span is None:
                    continue
                
                start_idx, end_idx = span
                
                # Get activations using hooks
                with torch_no_grad_and_cleanup(self.device):
                    # Run model to get residual stream
                    _, cache = self.model.run_with_cache(
                        tokens,
                        names_filter=[f"blocks.{layer_idx}.hook_resid_post"]
                    )
                    
                    # Get residual activations for the word span
                    residual = cache[f"blocks.{layer_idx}.hook_resid_post"][0, start_idx:end_idx]  # [seq_len, d_model]
                    
                    # Apply SAE to get latent activations
                    sae_acts = sae_model.encode(residual)  # [seq_len, d_sae]
                    
                    # Take max activation across the token span for each latent
                    max_acts = sae_acts.max(dim=0).values  # [d_sae]
                    
                    # Store activations
                    for latent_idx in range(max_acts.shape[0]):
                        if latent_idx not in latent_activations:
                            latent_activations[latent_idx] = []
                        latent_activations[latent_idx].append(max_acts[latent_idx].item())
                
                valid_samples += 1
                
                # Clear cache periodically
                if valid_samples % 500 == 0:
                    with torch_memory_cleanup(self.device):
                        pass
        
        logger.info(f"Extracted activations from {valid_samples} valid samples")
        return latent_activations
    
    def compute_pile_frequencies(self, layer_idx: int, sae_model) -> torch.Tensor:
        """
        Compute activation frequencies for all latents on Pile dataset.
        
        Args:
            layer_idx: Layer index
            sae_model: SAE model for the specified layer
            
        Returns:
            Tensor of shape [n_latents] with activation frequencies
        """
        # Check cache first
        cache_file = self.pile_cache_dir / f"layer_{layer_idx}_pile_frequencies.pt"
        metadata_file = self.pile_cache_dir / f"layer_{layer_idx}_metadata.json"
        
        if cache_file.exists() and metadata_file.exists():
            logger.info(f"Loading cached Pile frequencies for layer {layer_idx}")
            with torch_no_grad_and_cleanup(self.device):
                frequencies = torch.load(cache_file, map_location=self.device)
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded cache: {metadata['n_samples']} samples, computed on {metadata['timestamp']}")
            return frequencies
        
        # Load Pile dataset
        texts = self.load_pile_dataset()
        
        # Extract activations
        latent_activations = self.extract_word_activations(texts, layer_idx, sae_model)
        
        # Compute frequencies
        n_latents = sae_model.d_sae
        frequencies = torch.zeros(n_latents)
        
        for latent_idx in range(n_latents):
            if latent_idx in latent_activations:
                activations = torch.tensor(latent_activations[latent_idx])
                # Fraction of samples where activation > eps
                frequencies[latent_idx] = (activations > self.eps).float().mean()
        
        # Save cache with atomic write
        with atomic_file_write(str(cache_file), mode='wb') as f:
            torch.save(frequencies, f)
        
        metadata = {
            'n_samples': len(texts),
            'timestamp': datetime.now().isoformat(),
            'n_latents': n_latents,
            'layer_idx': layer_idx
        }
        with atomic_file_write(str(metadata_file), mode='w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Computed and cached Pile frequencies for {n_latents} latents")
        return frequencies
    
    def filter_by_pile_frequency(self,
                                separation_scores: SeparationScores,
                                pile_frequencies: torch.Tensor,
                                direction: str = 'correct',
                                pile_threshold: float = 0.02,
                                top_k: int = 20) -> Dict[str, Any]:
        """
        Filter top-k latents based on Pile activation frequency.
        
        Args:
            separation_scores: Separation scores for the layer
            pile_frequencies: Pile activation frequencies for all latents
            direction: 'correct' or 'incorrect' - which scores to use
            pile_threshold: Maximum allowed Pile activation frequency
            top_k: Maximum number of latents to return
            
        Returns:
            Dictionary with filtered latent information
        """
        # Get appropriate scores
        if direction == 'correct':
            scores = separation_scores.s_correct
        elif direction == 'incorrect':
            scores = separation_scores.s_incorrect
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'correct' or 'incorrect'")
        
        # Rank latents by separation score (highest first)
        sorted_indices = np.argsort(scores.cpu().numpy())[::-1].copy()
        
        # Filter by Pile frequency
        filtered_indices = []
        filtered_scores = []
        filtered_pile_freqs = []
        
        for idx in sorted_indices:
            # Check Pile frequency
            if pile_frequencies[idx] < pile_threshold:
                filtered_indices.append(idx)
                filtered_scores.append(scores[idx].item())
                filtered_pile_freqs.append(pile_frequencies[idx].item())
                
                # Stop when we have enough
                if len(filtered_indices) >= top_k:
                    break
        
        logger.info(f"Filtered {len(filtered_indices)} latents with Pile frequency < {pile_threshold}")
        
        return {
            'latent_indices': filtered_indices,
            'separation_scores': filtered_scores,
            'pile_frequencies': filtered_pile_freqs,
            'n_filtered': len(filtered_indices),
            'top_latent': filtered_indices[0] if filtered_indices else None,
            'direction': direction,
            'pile_threshold': pile_threshold
        }