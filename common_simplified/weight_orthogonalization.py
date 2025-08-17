"""
Gemma-2 specific weight orthogonalization implementation.

Handles the specific architecture and weight storage patterns of the Gemma-2 model,
including transposed weight matrices in HuggingFace implementations.
"""

import torch
from transformers import AutoModelForCausalLM
from common.weight_utils import get_orthogonalized_matrix, get_weight_change_magnitude
from common.logging import get_logger
from typing import Dict, List, Optional

logger = get_logger("weight_orthogonalization")


def orthogonalize_gemma_weights(
    model: AutoModelForCausalLM, 
    direction: torch.Tensor,
    target_weights: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Orthogonalize Gemma-2 model weights along PVA direction.
    
    This function modifies the model's weights in-place to remove the component
    aligned with a specific SAE decoder direction. This is a permanent modification
    that affects all future model outputs.
    
    Args:
        model: HuggingFace Gemma model to modify
        direction: PVA feature decoder direction [d_model]
        target_weights: List of weight types to orthogonalize
                       Options: 'embed', 'attn_o', 'mlp_down'
                       (default: ['embed', 'attn_o', 'mlp_down'])
    
    Returns:
        Dictionary mapping weight names to change magnitudes (Frobenius norms)
        
    Note:
        HuggingFace linear layers store weights transposed:
        - Mathematical operation: output = input @ W^T + bias
        - Storage format: W.shape = [out_features, in_features]
        - Therefore we transpose before orthogonalization and transpose back
    """
    if target_weights is None:
        target_weights = ['embed', 'attn_o', 'mlp_down']
    
    changes = {}
    
    # Ensure direction is on the correct device
    direction = direction.to(model.device)
    
    # Orthogonalize embedding weights
    if 'embed' in target_weights:
        try:
            # Embedding weights: [vocab_size, d_model]
            # No transposition needed as embeddings map tokens to vectors directly
            original = model.model.embed_tokens.weight.data.clone()
            model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
                model.model.embed_tokens.weight.data, direction
            )
            change = get_weight_change_magnitude(
                original, model.model.embed_tokens.weight.data
            )
            changes['embedding'] = change
            logger.info(f"Orthogonalized embeddings, change magnitude: {change:.4f}")
        except Exception as e:
            logger.error(f"Failed to orthogonalize embeddings: {e}")
    
    # Process each transformer layer
    for i, block in enumerate(model.model.layers):
        # Attention output projection
        if 'attn_o' in target_weights:
            try:
                # o_proj weight shape: [d_model, d_model] (transposed storage)
                # We need to orthogonalize columns (output dimensions)
                original = block.self_attn.o_proj.weight.data.clone()
                
                # Transpose, orthogonalize, transpose back
                block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
                    block.self_attn.o_proj.weight.data.T, direction
                ).T
                
                change = get_weight_change_magnitude(
                    original, block.self_attn.o_proj.weight.data
                )
                changes[f'layer_{i}_attn_o'] = change
                
                if i == 0:  # Log details for first layer only to avoid spam
                    logger.debug(f"Layer {i} attention output projection: change {change:.4f}")
                    
            except Exception as e:
                logger.error(f"Failed to orthogonalize layer {i} attention: {e}")
        
        # MLP down projection
        if 'mlp_down' in target_weights:
            try:
                # down_proj weight shape: [d_model, d_intermediate] (transposed storage)
                # We need to orthogonalize columns (output dimensions going to residual stream)
                original = block.mlp.down_proj.weight.data.clone()
                
                # Transpose, orthogonalize, transpose back
                block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
                    block.mlp.down_proj.weight.data.T, direction
                ).T
                
                change = get_weight_change_magnitude(
                    original, block.mlp.down_proj.weight.data
                )
                changes[f'layer_{i}_mlp_down'] = change
                
                if i == 0:  # Log details for first layer only
                    logger.debug(f"Layer {i} MLP down projection: change {change:.4f}")
                    
            except Exception as e:
                logger.error(f"Failed to orthogonalize layer {i} MLP: {e}")
    
    # Calculate and log total change
    total_change = sum(changes.values())
    logger.info(f"Total weight change magnitude: {total_change:.4f}")
    logger.info(f"Modified {len(changes)} weight matrices")
    
    # Log per-component average changes
    embedding_change = changes.get('embedding', 0.0)
    attn_changes = [v for k, v in changes.items() if 'attn_o' in k]
    mlp_changes = [v for k, v in changes.items() if 'mlp_down' in k]
    
    if attn_changes:
        logger.info(f"Average attention change: {sum(attn_changes)/len(attn_changes):.4f}")
    if mlp_changes:
        logger.info(f"Average MLP change: {sum(mlp_changes)/len(mlp_changes):.4f}")
    
    return changes


def create_orthogonalized_model(
    model_name: str,
    direction: torch.Tensor,
    target_weights: Optional[List[str]] = None,
    device: Optional[str] = None
) -> tuple[AutoModelForCausalLM, Dict[str, float]]:
    """
    Create a fresh model with orthogonalized weights.
    
    This is a convenience function that loads a model and immediately
    orthogonalizes it, useful for experiments that need fresh models.
    
    Args:
        model_name: HuggingFace model identifier
        direction: PVA feature decoder direction to remove
        target_weights: Weight types to orthogonalize
        device: Device to load model on
        
    Returns:
        Tuple of (orthogonalized model, weight change dictionary)
    """
    from common_simplified.model_loader import load_model_and_tokenizer
    
    logger.info(f"Loading fresh model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    
    logger.info("Applying weight orthogonalization")
    changes = orthogonalize_gemma_weights(model, direction, target_weights)
    
    return model, changes