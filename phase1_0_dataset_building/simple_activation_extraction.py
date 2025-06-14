"""
Simple activation extraction for Phase 1 following KISS principle.

This module shows how to extract activations during generation without
complex abstraction layers.
"""

import torch
from typing import List, Optional
from pathlib import Path
import numpy as np


def extract_activation_during_generation(
    model, 
    inputs, 
    layer_idx: int, 
    position: int = -1
) -> torch.Tensor:
    """
    Extract activation from specific layer during model forward pass.
    
    Args:
        model: HuggingFace model
        inputs: Tokenized inputs dict
        layer_idx: Layer to extract from
        position: Token position (-1 for last)
        
    Returns:
        Extracted activation tensor
    """
    activation = None
    
    def extraction_hook(module, input, output):
        nonlocal activation
        # Handle Gemma-2 output format
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        # Extract specific position
        if position == -1:
            activation = hidden_states[:, -1, :].detach().cpu()
        else:
            activation = hidden_states[:, position, :].detach().cpu()
    
    # Register hook on the specific layer
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Gemma/LLaMA style
        layer = model.model.layers[layer_idx]
    else:
        raise ValueError("Unknown model architecture")
    
    handle = layer.register_forward_hook(extraction_hook)
    
    try:
        # Run forward pass (this triggers the hook)
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        # Always clean up hook
        handle.remove()
    
    return activation


def save_activations_simple(
    activations: torch.Tensor,
    task_id: str,
    layer_idx: int,
    is_correct: bool,
    output_dir: Path
) -> None:
    """
    Save activations to disk in a simple format.
    
    Args:
        activations: Activation tensor to save
        task_id: Task identifier
        layer_idx: Layer index
        is_correct: Whether this is from correct code
        output_dir: Base output directory
    """
    # Determine subdirectory
    subdir = "correct" if is_correct else "incorrect"
    save_dir = output_dir / "activations" / subdir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as compressed numpy
    filepath = save_dir / f"{task_id}_layer_{layer_idx}.npz"
    np.savez_compressed(
        filepath,
        activations=activations.numpy(),
        layer=layer_idx,
        task_id=task_id
    )


# Example usage in dataset building:
def process_task_with_activations(
    model,
    tokenizer,
    task_id: str,
    prompt: str,
    layers: List[int],
    output_dir: Path
) -> dict:
    """
    Generate code and extract activations in one pass.
    
    This is what Phase 1 should do - generate code and save
    activations at the same time to avoid duplicate inference.
    """
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate code
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True
        )
    
    # Decode generated code
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Now extract activations for the generated sequence
    # Re-tokenize the full generated text
    full_text = generated_code  # This includes prompt + generated
    full_inputs = tokenizer(full_text, return_tensors="pt", truncation=True)
    full_inputs = {k: v.to(model.device) for k, v in full_inputs.items()}
    
    # Extract activations for each layer
    activations = {}
    for layer_idx in layers:
        act = extract_activation_during_generation(
            model, full_inputs, layer_idx, position=-1
        )
        activations[layer_idx] = act
    
    # Test the code to determine if correct
    # ... (testing logic here) ...
    is_correct = True  # Placeholder
    
    # Save all activations
    for layer_idx, act in activations.items():
        save_activations_simple(
            act, task_id, layer_idx, is_correct, output_dir
        )
    
    return {
        'task_id': task_id,
        'generated_code': generated_code,
        'is_correct': is_correct,
        'layers_extracted': layers
    }