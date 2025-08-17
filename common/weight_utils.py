"""
Weight manipulation utilities for neural network orthogonalization.

Provides functions for projecting out specific directions from weight matrices,
used in Phase 5.3 for permanent PVA feature removal.
"""

import torch
import einops
from torch import Tensor, FloatTensor
from typing import Optional


def get_orthogonalized_matrix(matrix: FloatTensor, vec: FloatTensor) -> FloatTensor:
    """
    Remove projection of matrix rows onto direction vector.
    
    This function orthogonalizes a weight matrix with respect to a given direction,
    effectively removing the component of each weight vector that aligns with the
    target direction while preserving all orthogonal components.
    
    Args:
        matrix: Weight matrix to orthogonalize [..., d_model]
        vec: Direction to project out [d_model]
    
    Returns:
        Orthogonalized matrix with same shape as input
        
    Mathematical formula:
        W_orthogonalized = W - ((W @ d) / ||d||²) × d^T
    """
    # Normalize direction vector to unit length for numerical stability
    vec = vec / torch.norm(vec)
    
    # Match device and dtype of the matrix
    vec = vec.to(matrix.device).to(matrix.dtype)
    
    # Compute projection of matrix onto direction
    # Using einsum for clarity and efficiency
    proj = einops.einsum(
        matrix, vec.unsqueeze(-1), 
        '... d_model, d_model single -> ... single'
    ) * vec
    
    # Subtract projection to get orthogonalized matrix
    return matrix - proj


def get_weight_change_magnitude(original: FloatTensor, modified: FloatTensor) -> float:
    """
    Calculate the Frobenius norm of weight changes.
    
    Args:
        original: Original weight matrix
        modified: Modified weight matrix
        
    Returns:
        Frobenius norm of the difference
    """
    return torch.norm(modified - original).item()


def verify_orthogonalization(
    matrix: FloatTensor, 
    direction: FloatTensor,
    tolerance: float = 1e-6
) -> bool:
    """
    Verify that a matrix has been successfully orthogonalized to a direction.
    
    After orthogonalization, the projection of any row of the matrix onto
    the direction should be near zero.
    
    Args:
        matrix: Supposedly orthogonalized matrix [..., d_model]
        direction: Direction that should have been removed [d_model]
        tolerance: Maximum allowed projection magnitude
        
    Returns:
        True if orthogonalization is successful (projections near zero)
    """
    # Normalize direction
    direction = direction / torch.norm(direction)
    direction = direction.to(matrix.device).to(matrix.dtype)
    
    # Compute projections of all rows onto direction
    projections = torch.matmul(matrix, direction)
    
    # Check if all projections are near zero
    max_projection = torch.max(torch.abs(projections)).item()
    
    return max_projection < tolerance