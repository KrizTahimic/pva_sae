"""
Phase 2.2 Runner: Cache pile dataset activations for filtering.

This module processes the NeelNanda/pile-10k dataset to extract activations
at random word positions, establishing a baseline for general language features.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional
import random
from datasets import load_dataset

from common.config import Config
from common.logging import get_logger, tqdm_with_logging
from common.phase_discovery import get_phase_output_dir, filter_by_range, get_dataset_range
from common.model_loader import load_model_and_tokenizer
from common.tensor_utils import save_activation
from .pile_activation_hook import PileActivationHook
from .utils import find_word_position, validate_pile_sample

# Module logger
logger = get_logger("pile_caching", phase="2.2")

def run_phase2_2_caching(config: Config, gpu_id: int = 0, n_gpus: int = 1, device: str = "cuda") -> None:
    """
    Cache pile dataset activations for filtering.

    Processes texts one at a time for simplicity (KISS principle).
    Supports multi-GPU via index-based work splitting.

    Args:
        config: Configuration object
        gpu_id: GPU worker ID (0-indexed) for parallel execution
        n_gpus: Total number of GPUs for parallel execution
        device: Device to use for model
    """
    # Setup output directory (uses model/dataset-aware path)
    output_dir = Path(get_phase_output_dir("2.2", config)) / "pile_activations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {config.pile_samples} pile samples")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {config.model_name}")
    model, tokenizer = load_model_and_tokenizer(config.model_name, device)
    model.eval()
    
    # Load pile-10k dataset
    logger.info("Loading pile-10k dataset...")
    dataset = load_dataset("NeelNanda/pile-10k", split='train')
    texts = dataset['text'][:config.pile_samples]
    
    # Pre-select random words from each text (seeded for reproducibility)
    random.seed(42)
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
    # Get start_idx for filename tracking before filtering
    start_idx, end_idx = get_dataset_range(config, len(texts))
    texts = filter_by_range(texts, config, "pile samples")
    substrings = substrings[start_idx:start_idx + len(texts)]

    # Filter texts by GPU (round-robin distribution)
    if n_gpus > 1:
        from common.parallel_runner import _get_gpu_task_indices
        gpu_indices = _get_gpu_task_indices(len(texts), n_gpus, gpu_id)
        texts = [texts[i] for i in gpu_indices]
        substrings = [substrings[i] for i in gpu_indices]
        # Store original indices for filename generation
        original_indices = [start_idx + i for i in gpu_indices]
        # Update end_idx for logging
        end_idx = start_idx + len(texts)
        logger.info(f"GPU {gpu_id}/{n_gpus}: Processing {len(texts)} samples")
    else:
        # Single GPU mode: sequential indices
        original_indices = list(range(start_idx, start_idx + len(texts)))

    # Process each text individually
    processed_count = 0
    skipped_count = 0
    checkpoint_count = 0

    # Progress bar - use enumerate with start offset for correct filenames
    for local_idx, (text, random_word) in tqdm_with_logging(
        enumerate(zip(texts, substrings)), logger, desc="Processing pile samples", total=len(texts)
    ):
        idx = original_indices[local_idx]  # Original index for filename

        # Checkpointing: skip if all layer files already exist for this sample
        if n_gpus > 1:
            first_layer_path = output_dir / f"gpu{gpu_id}_{idx}_layer_{config.activation_layers[0]}.safetensors"
        else:
            first_layer_path = output_dir / f"{idx}_layer_{config.activation_layers[0]}.safetensors"
        if first_layer_path.exists():
            checkpoint_count += 1
            continue

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
                    
                    # Save activation if extracted (preserves bfloat16)
                    if hook.activation is not None:
                        if n_gpus > 1:
                            save_path = output_dir / f"gpu{gpu_id}_{idx}_layer_{layer_idx}.safetensors"
                        else:
                            save_path = output_dir / f"{idx}_layer_{layer_idx}.safetensors"
                        save_activation(hook.activation, save_path)
                finally:
                    # Always remove hook
                    handle.remove()
        
        processed_count += 1
        
        # Periodic memory cleanup
        if processed_count % 100 == 0:
            torch.cuda.empty_cache()
            logger.info(f"Processed {processed_count} texts from range [{start_idx}, {end_idx})")
    
    # Final cleanup
    torch.cuda.empty_cache()

    logger.info(f"Completed: {processed_count} processed, {skipped_count} skipped, {checkpoint_count} checkpointed from range [{start_idx}, {end_idx})")
    logger.info(f"Activations saved to: {output_dir}")

    # Write phase_output.json manifest (skip in parallel mode - orchestrator writes combined manifest)
    if n_gpus == 1:
        from common.phase_discovery import write_phase_output

        phase_output_dir = output_dir.parent  # phase2_2 dir, not pile_activations subdir
        write_phase_output(
            phase="2.2",
            outputs={
                "primary": "pile_activations/",
            },
            config=config,
            output_dir=str(phase_output_dir),
            config_keys=['model_name', 'pile_samples', 'activation_layers']
        )
        logger.info(f"Saved phase_output.json manifest to {phase_output_dir}")