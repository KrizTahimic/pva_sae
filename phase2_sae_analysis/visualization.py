"""
Visualization utilities for SAE analysis results.
"""

import matplotlib.pyplot as plt
import torch
from typing import List, Dict, Optional


def plot_separation_scores(
    scores,
    top_k: int = 10,
    figsize: tuple = (12, 5)
) -> None:
    """
    Plot top separation scores for both correct and incorrect directions.
    
    Args:
        scores: SeparationScores object from sae_analyzer
        top_k: Number of top features to show
        figsize: Figure size
    """
    # Get top features
    top_correct_scores, top_correct_idx = torch.topk(scores.s_correct, top_k)
    top_incorrect_scores, top_incorrect_idx = torch.topk(scores.s_incorrect, top_k)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Correct direction
    ax1.bar(range(top_k), top_correct_scores.cpu().numpy(), color='green', alpha=0.7)
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Separation Score (f_correct - f_incorrect)')
    ax1.set_title(f'Top {top_k} Correct Code Features')
    ax1.set_xticks(range(top_k))
    ax1.set_xticklabels([f'{idx}' for idx in top_correct_idx.cpu().numpy()], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Incorrect direction
    ax2.bar(range(top_k), top_incorrect_scores.cpu().numpy(), color='red', alpha=0.7)
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Separation Score (f_incorrect - f_correct)')
    ax2.set_title(f'Top {top_k} Incorrect Code Features')
    ax2.set_xticks(range(top_k))
    ax2.set_xticklabels([f'{idx}' for idx in top_incorrect_idx.cpu().numpy()], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_activation_fractions(
    scores,
    feature_idx: int,
    title: Optional[str] = None
) -> None:
    """
    Plot activation fractions for a specific feature.
    
    Args:
        scores: SeparationScores object
        feature_idx: Feature index to visualize
        title: Optional plot title
    """
    f_correct = scores.f_correct[feature_idx].item()
    f_incorrect = scores.f_incorrect[feature_idx].item()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    bars = ax.bar(['Correct', 'Incorrect'], [f_correct, f_incorrect], 
                   color=['green', 'red'], alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, [f_correct, f_incorrect]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    
    ax.set_ylabel('Activation Fraction')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    if title is None:
        title = f'Feature {feature_idx} Activation Fractions'
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()


def summarize_pva_directions(correct_dir, incorrect_dir) -> None:
    """
    Print a formatted summary of PVA latent directions.
    
    Args:
        correct_dir: PVALatentDirection for correct code
        incorrect_dir: PVALatentDirection for incorrect code
    """
    print("=" * 60)
    print("PROGRAM VALIDITY AWARENESS LATENT DIRECTIONS")
    print("=" * 60)
    
    print(f"\n{correct_dir}")
    print(f"  Activation pattern:")
    print(f"    - On correct code: {correct_dir.f_correct:.1%}")
    print(f"    - On incorrect code: {correct_dir.f_incorrect:.1%}")
    print(f"    - Difference: {correct_dir.separation_score:+.3f}")
    
    print(f"\n{incorrect_dir}")
    print(f"  Activation pattern:")
    print(f"    - On correct code: {incorrect_dir.f_correct:.1%}")
    print(f"    - On incorrect code: {incorrect_dir.f_incorrect:.1%}")
    print(f"    - Difference: {incorrect_dir.separation_score:+.3f}")
    
    print("\n" + "=" * 60)