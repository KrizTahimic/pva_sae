"""
Phase 2.2 Runner: Cache pile dataset activations for filtering.

This module processes the NeelNanda/pile-10k dataset to extract activations
at random word positions, establishing a baseline for general language features.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import random
from tqdm import tqdm
from datasets import load_dataset

from common.config import Config
from common.logging import get_logger
from common_simplified.model_loader import load_model_and_tokenizer
from .pile_activation_hook import PileActivationHook
from .utils import find_word_position, validate_pile_sample

# Module logger
logger = get_logger("pile_caching", phase="2.2")


def run_phase2_2_caching(config: Config, device: str = "cuda") -> None:
    """
    Cache pile dataset activations for filtering.
    
    Processes texts one at a time for simplicity (KISS principle).
    Supports multi-GPU via index-based work splitting.
    
    Args:
        config: Configuration object
        device: Device to use for model
    """
    # Setup output directory
    output_dir = Path(config.get_phase_output_dir("2.2")) / "pile_activations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get run count from config (defaults to pile_samples)
    run_count = getattr(config, '_run_count', config.pile_samples)
    logger.info(f"Processing {run_count} pile samples")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {config.model_name}")
    model, tokenizer = load_model_and_tokenizer(config.model_name, device)
    model.eval()
    
    # Load pile-10k dataset
    logger.info("Loading pile-10k dataset...")
    dataset = load_dataset("NeelNanda/pile-10k", split='train')
    texts = dataset['text'][:run_count]  # Use specified count
    
    # Pre-select random words from each text
    logger.info("Selecting random words from texts...")
    substrings = []
    for text in texts:
        words = text.split()
        if words:
            # Select a random word
            word = random.choice(words)
            # Clean punctuation from word edges for better matching
            word = word.strip('.,!?;:"\'')
            substrings.append(word)
        else:
            substrings.append(None)  # Handle empty texts
    
    # Handle start/end indices for multi-GPU processing
    start_idx = getattr(config, 'dataset_start_idx', 0)
    end_idx = getattr(config, 'dataset_end_idx', None) or len(texts)
    end_idx = min(end_idx, len(texts))  # Ensure we don't exceed dataset size
    
    logger.info(f"Processing pile samples {start_idx} to {end_idx-1} one at a time...")
    
    # Process each text individually
    processed_count = 0
    skipped_count = 0
    
    # Progress bar for current GPU's work
    pbar = tqdm(range(start_idx, end_idx), desc="Processing pile samples")
    
    for idx in pbar:
        text = texts[idx]
        random_word = substrings[idx]
        
        if random_word is None:
            skipped_count += 1
            continue
            
        # Validate sample
        if not validate_pile_sample(text, random_word):
            skipped_count += 1
            continue
            
        # Step 1: Tokenize and check if random word survives truncation
        inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
        truncated_text = tokenizer.decode(inputs.input_ids[0])
        
        if random_word.lower() not in truncated_text.lower():
            skipped_count += 1
            continue
        
        # Step 2: Find position of random word
        position = find_word_position(random_word, inputs.input_ids[0], tokenizer)
        if position is None:
            skipped_count += 1
            continue
        
        # Step 3: Extract activation at that position for each layer
        inputs = inputs.to(device)
        
        with torch.no_grad():
            for layer_idx in config.activation_layers:
                # Create hook for this specific position
                hook = PileActivationHook(position)
                
                # Register hook on the appropriate layer
                if hasattr(model, 'model'):  # Gemma structure
                    handle = model.model.layers[layer_idx].register_forward_hook(hook.hook_fn)
                else:
                    handle = model.layers[layer_idx].register_forward_hook(hook.hook_fn)
                
                try:
                    # Run forward pass
                    _ = model(inputs.input_ids)
                    
                    # Save activation if extracted
                    if hook.activation is not None:
                        save_path = output_dir / f"{idx}_layer_{layer_idx}.npz"
                        np.savez_compressed(save_path, activation=hook.activation.numpy())
                finally:
                    # Always remove hook
                    handle.remove()
        
        processed_count += 1
        
        # Update progress bar description
        pbar.set_postfix({
            'processed': processed_count,
            'skipped': skipped_count,
            'word': random_word[:20] + '...' if len(random_word) > 20 else random_word
        })
        
        # Periodic memory cleanup
        if processed_count % 100 == 0:
            torch.cuda.empty_cache()
            logger.info(f"Processed {processed_count} texts from range [{start_idx}, {end_idx})")
    
    # Final cleanup
    torch.cuda.empty_cache()
    
    logger.info(f"Completed: {processed_count} processed, {skipped_count} skipped from range [{start_idx}, {end_idx})")
    logger.info(f"Activations saved to: {output_dir}")