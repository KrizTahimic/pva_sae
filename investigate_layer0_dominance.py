"""Investigate why all top features come from layer 0."""

import json
import numpy as np
from pathlib import Path
import glob

def investigate_issue():
    """Analyze possible causes for layer 0 dominance."""
    
    print("=== Investigating Layer 0 Dominance Issue ===\n")
    
    # 1. Check activation files availability
    print("1. CHECKING ACTIVATION FILE AVAILABILITY:")
    phase1_dir = Path("data/phase1_0/activations")
    if phase1_dir.exists():
        for category in ["correct", "incorrect"]:
            cat_dir = phase1_dir / category
            if cat_dir.exists():
                print(f"\n{category.upper()} activations:")
                for layer in [0, 6, 8, 15, 17]:
                    layer_files = list(cat_dir.glob(f"*_layer_{layer}.npz"))
                    print(f"  Layer {layer}: {len(layer_files)} files")
                    
                    # Check if files are empty or corrupted
                    if layer_files:
                        sample_file = layer_files[0]
                        try:
                            data = np.load(sample_file)
                            shape = data[data.files[0]].shape
                            print(f"    Sample shape: {shape}")
                        except Exception as e:
                            print(f"    ERROR loading: {e}")
    
    # 2. Check Phase 2.10 results
    print("\n2. PHASE 2.10 RESULTS ANALYSIS:")
    results_file = Path("data/phase2_10/sae_analysis_results.json")
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"Analyzed layers: {results['activation_layers']}")
        
        # Check per-layer results
        for layer in results['activation_layers']:
            layer_file = Path(f"data/phase2_10/layer_{layer}_features.json")
            if layer_file.exists():
                with open(layer_file, 'r') as f:
                    layer_data = json.load(f)
                
                # Get top t-statistics for this layer
                correct_top = layer_data['features']['correct'][:3] if 'features' in layer_data else []
                incorrect_top = layer_data['features']['incorrect'][:3] if 'features' in layer_data else []
                
                print(f"\nLayer {layer}:")
                print(f"  n_correct: {layer_data.get('n_correct', 'N/A')}")
                print(f"  n_incorrect: {layer_data.get('n_incorrect', 'N/A')}")
                
                if correct_top:
                    print(f"  Top correct t-stat: {correct_top[0]['t_statistic']:.3f}")
                else:
                    print(f"  Top correct t-stat: No features found")
                    
                if incorrect_top:
                    print(f"  Top incorrect t-stat: {incorrect_top[0]['t_statistic']:.3f}")
                else:
                    print(f"  Top incorrect t-stat: No features found")
    
    print("\n" + "="*60)
    print("POSSIBLE CAUSES FOR LAYER 0 DOMINANCE:\n")
    
    causes = [
        ("Data Issue", "Phase 1 may have only saved layer 0 activations properly"),
        ("SAE Loading", "GemmaScope SAEs might not be available/loading for other layers"),
        ("Activation Shape", "Other layers might have different tensor shapes causing issues"),
        ("File Naming", "Activation files for other layers might have different naming patterns"),
        ("Silent Failures", "Other layers might be failing without proper error reporting"),
        ("Statistical Reality", "Layer 0 might genuinely have higher separation (unlikely)"),
        ("Zero Activations", "Other layers might have mostly zero/dead features"),
        ("Memory Issues", "Large tensor operations might be failing for deeper layers"),
        ("Config Issue", "activation_layers config might not be used correctly"),
        ("Early Exit", "Code might stop after processing layer 0")
    ]
    
    for i, (cause, description) in enumerate(causes, 1):
        print(f"{i}. {cause}:")
        print(f"   {description}\n")
    
    print("RECOMMENDED DEBUGGING STEPS:")
    print("1. Check if Phase 1 saved all layer activations correctly")
    print("2. Verify SAE models load for all layers (not just layer 0)")
    print("3. Add detailed logging in analyze_layer() method")
    print("4. Check if compute_t_statistics() is called for all layers")
    print("5. Verify tensor shapes are consistent across layers")

if __name__ == "__main__":
    investigate_issue()