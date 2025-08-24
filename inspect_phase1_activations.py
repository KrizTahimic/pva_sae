"""Inspect Phase 1 activations to diagnose the issue."""

import numpy as np
import torch
from pathlib import Path

def inspect_activations():
    """Look at actual Phase 1 activation values."""
    
    print("=== Inspecting Phase 1 Activations ===\n")
    
    phase1_dir = Path("data/phase1_0/activations")
    
    for category in ["correct", "incorrect"]:
        print(f"{category.upper()} activations:")
        cat_dir = phase1_dir / category
        
        for layer in [0, 6, 8]:  # Just check first 3 layers
            layer_files = list(cat_dir.glob(f"*_layer_{layer}.npz"))
            if not layer_files:
                print(f"  Layer {layer}: No files found")
                continue
                
            print(f"  Layer {layer}: {len(layer_files)} files")
            
            # Load first few files and inspect
            for i, file_path in enumerate(layer_files[:3]):
                try:
                    data = np.load(file_path)
                    activation = data[data.files[0]]
                    
                    print(f"    File {i+1} ({file_path.name}):")
                    print(f"      Shape: {activation.shape}")
                    print(f"      Mean: {activation.mean():.6f}")
                    print(f"      Std: {activation.std():.6f}")
                    print(f"      Min: {activation.min():.6f}")
                    print(f"      Max: {activation.max():.6f}")
                    print(f"      Non-zero: {(activation != 0).sum()}/{activation.size}")
                    
                    # Check if all values are the same
                    unique_vals = np.unique(activation)
                    if len(unique_vals) <= 5:  # Very few unique values
                        print(f"      Unique values: {unique_vals}")
                    else:
                        print(f"      Unique values: {len(unique_vals)} (diverse)")
                    
                    # Sample some values
                    flat = activation.flatten()
                    sample_indices = np.linspace(0, len(flat)-1, 10, dtype=int)
                    sample_vals = flat[sample_indices]
                    print(f"      Sample values: {sample_vals}")
                    print()
                    
                except Exception as e:
                    print(f"    ERROR loading {file_path.name}: {e}")
            
        print()

if __name__ == "__main__":
    inspect_activations()