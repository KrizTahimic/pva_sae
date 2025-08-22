#!/usr/bin/env python3
"""Test activation consistency at temperature=0 across multiple runs."""

import torch
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from common_simplified.activation_hooks import ActivationExtractor
from common.prompt_utils import PromptBuilder
import matplotlib.pyplot as plt
import seaborn as sns

def run_single_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer_idx: int,
    device: str = "cuda"
) -> Tuple[str, np.ndarray]:
    """Run a single generation and capture activations."""
    
    # Setup activation extractor
    extractor = ActivationExtractor(model, [layer_idx], position=-1)
    extractor.setup_hooks()
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000)
    
    # Move inputs to the same device as the model
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    # Generate with temperature=0 (greedy decoding)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,  # Deterministic
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Get generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Get activations
    activations = extractor.get_activations()[layer_idx]
    
    # Clean up hooks
    extractor.remove_hooks()
    
    return generated_text, activations.cpu().numpy()

def analyze_activation_differences(activations_list: List[np.ndarray]) -> Dict:
    """Analyze differences between multiple activation arrays."""
    
    n_runs = len(activations_list)
    
    # Stack all activations
    stacked = np.stack(activations_list, axis=0)  # Shape: (n_runs, seq_len, hidden_dim)
    
    # Calculate statistics
    mean_act = np.mean(stacked, axis=0)
    std_act = np.std(stacked, axis=0)
    
    # Calculate pairwise differences
    pairwise_diffs = []
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            diff = np.abs(activations_list[i] - activations_list[j])
            pairwise_diffs.append({
                'run_i': i,
                'run_j': j,
                'mean_diff': np.mean(diff),
                'max_diff': np.max(diff),
                'min_diff': np.min(diff),
                'std_diff': np.std(diff),
                'percent_nonzero': np.mean(diff > 1e-8) * 100
            })
    
    # Check if outputs are identical
    all_identical = all(
        np.allclose(activations_list[0], activations_list[i], rtol=1e-7, atol=1e-8)
        for i in range(1, n_runs)
    )
    
    return {
        'n_runs': n_runs,
        'shape': stacked.shape,
        'mean_std': np.mean(std_act),
        'max_std': np.max(std_act),
        'min_std': np.min(std_act),
        'all_identical': all_identical,
        'pairwise_differences': pairwise_diffs,
        'percent_varying_positions': np.mean(std_act > 1e-8) * 100
    }

def main():
    """Run activation consistency test."""
    
    print("Loading model and tokenizer...")
    # Check for available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    model_name = "google/gemma-2-2b"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with proper device handling
    if device == "mps":
        # For MPS, load to CPU first then move
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        if device == "cpu":
            model = model.to(device)
    
    model.eval()
    
    # Test parameters
    n_runs = 5  # Number of times to run the same prompt
    layer_idx = 8  # Layer to extract activations from
    
    # Example MBPP problem
    problem = {
        "task_id": "Mbpp/2",
        "text": "Write a function to find the similar elements from the given two tuple lists.",
        "test_list": [
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
            "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"
        ]
    }
    
    # Format prompt
    test_cases_str = "\n".join(problem["test_list"])
    prompt = PromptBuilder.build_prompt(problem["text"], test_cases_str)
    
    print(f"\nRunning {n_runs} generations with temperature=0...")
    print(f"Extracting activations from layer {layer_idx}")
    print("-" * 50)
    
    # Run multiple generations
    generations = []
    activations = []
    
    for i in range(n_runs):
        print(f"Run {i+1}/{n_runs}...")
        gen_text, act = run_single_generation(model, tokenizer, prompt, layer_idx, device)
        generations.append(gen_text)
        activations.append(act)
    
    # Check if generated texts are identical
    print("\n" + "="*50)
    print("GENERATION CONSISTENCY:")
    print("="*50)
    
    all_texts_identical = all(gen == generations[0] for gen in generations[1:])
    print(f"All generated texts identical: {all_texts_identical}")
    
    if not all_texts_identical:
        print("\nDifferent generations found:")
        for i, gen in enumerate(generations):
            print(f"\nRun {i+1}:")
            print(gen[:200] + "..." if len(gen) > 200 else gen)
    
    # Analyze activation differences
    print("\n" + "="*50)
    print("ACTIVATION ANALYSIS:")
    print("="*50)
    
    analysis = analyze_activation_differences(activations)
    
    print(f"Number of runs: {analysis['n_runs']}")
    print(f"Activation shape: {analysis['shape']}")
    print(f"All activations identical: {analysis['all_identical']}")
    print(f"Percentage of positions with variance > 1e-8: {analysis['percent_varying_positions']:.6f}%")
    print(f"Mean standard deviation: {analysis['mean_std']:.2e}")
    print(f"Max standard deviation: {analysis['max_std']:.2e}")
    print(f"Min standard deviation: {analysis['min_std']:.2e}")
    
    print("\nPairwise differences:")
    for diff in analysis['pairwise_differences']:
        print(f"  Run {diff['run_i']} vs Run {diff['run_j']}:")
        print(f"    Mean diff: {diff['mean_diff']:.2e}")
        print(f"    Max diff: {diff['max_diff']:.2e}")
        print(f"    % non-zero: {diff['percent_nonzero']:.6f}%")
    
    # Visualize variance
    if not analysis['all_identical']:
        print("\nCreating variance visualization...")
        
        stacked = np.stack(activations, axis=0)
        std_map = np.std(stacked, axis=0)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Heatmap of standard deviation
        im1 = axes[0].imshow(std_map.T, aspect='auto', cmap='hot')
        axes[0].set_title('Standard Deviation of Activations Across Runs')
        axes[0].set_xlabel('Token Position')
        axes[0].set_ylabel('Hidden Dimension')
        plt.colorbar(im1, ax=axes[0])
        
        # Line plot of mean std per position
        mean_std_per_pos = np.mean(std_map, axis=1)
        axes[1].plot(mean_std_per_pos)
        axes[1].set_title('Mean Standard Deviation per Token Position')
        axes[1].set_xlabel('Token Position')
        axes[1].set_ylabel('Mean Std Dev')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('activation_variance_temp0.png', dpi=150)
        print("Saved visualization to activation_variance_temp0.png")
    
    # Save results
    results_file = 'activation_consistency_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'all_texts_identical': all_texts_identical,
            'all_activations_identical': analysis['all_identical'],
            'percent_varying_positions': analysis['percent_varying_positions'],
            'mean_std': float(analysis['mean_std']),
            'max_std': float(analysis['max_std']),
            'n_runs': n_runs,
            'temperature': 0.0,
            'layer_idx': layer_idx
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()