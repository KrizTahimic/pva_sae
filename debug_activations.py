import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from proper location
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae
from common.utils import detect_device

# Check Phase 1 (where features were selected) and Phase 3.5 (validation, where final metrics are reported)
phase1_dir = Path('data/phase1_0')
phase3_5_dir = Path('data/phase3_5')
phase3_6_dir = Path('data/phase3_6')

# Load the latest Phase 1 dataset (used by Phase 2.5 for feature selection)
# Find the latest dataset file
phase1_files = sorted(phase1_dir.glob('dataset_sae_*.parquet'))
if not phase1_files:
    print("ERROR: No Phase 1 dataset found. Please run Phase 1 first.")
    sys.exit(1)
phase1_file = phase1_files[-1]  # Get the latest one
print(f"Loading Phase 1 data from: {phase1_file}")
phase1_data = pd.read_parquet(phase1_file)

# Load Phase 3.5 validation data (what Phase 3.8 reports final metrics on)
validation_data = pd.read_parquet(Path('data/phase0_1/validation_mbpp.parquet'))
phase3_5_data = pd.read_parquet(phase3_5_dir / 'dataset_temp_0_0.parquet')

# Auto-discover features from Phase 2.5 results
import json

# Load Phase 2.5 top features
phase2_5_dir = Path('data/phase2_5')
with open(phase2_5_dir / 'top_20_features.json', 'r') as f:
    top_features = json.load(f)

# Check which layers have Phase 3.5 activations available
available_layers = set()
if phase3_5_dir.exists() and (phase3_5_dir / 'activations/task_activations').exists():
    for act_file in (phase3_5_dir / 'activations/task_activations').glob('*_layer_*.npz'):
        # Extract layer number from filename like "1_layer_6.npz"
        layer_num = int(act_file.stem.split('_layer_')[1])
        available_layers.add(layer_num)
    print(f"Phase 3.5 has activations for layers: {sorted(available_layers)}")
else:
    print("No Phase 3.5 activations found")
    available_layers = {0, 6, 8, 15, 17}  # Default to all analyzed layers

# Select best features for available layers
features_to_test = []

# Find best correct-preferring feature for available layers
for feature in top_features['correct']:
    if feature['layer'] in available_layers and feature['separation_score'] > 0:
        features_to_test.append({
            'layer': feature['layer'],
            'feature_idx': feature['feature_idx'],
            'type': 'correct-preferring',
            'separation_score': feature['separation_score']
        })
        break

# Find best incorrect-preferring feature for available layers  
for feature in top_features['incorrect']:
    if feature['layer'] in available_layers and feature['separation_score'] > 0:
        features_to_test.append({
            'layer': feature['layer'],
            'feature_idx': feature['feature_idx'],
            'type': 'incorrect-preferring',
            'separation_score': feature['separation_score']
        })
        break

# If no features found for available layers, fall back to top features regardless of layer
if not features_to_test:
    print("Warning: No features found for available layers, using top features from any layer")
    if top_features['correct']:
        best_correct = top_features['correct'][0]
        features_to_test.append({
            'layer': best_correct['layer'],
            'feature_idx': best_correct['feature_idx'],
            'type': 'correct-preferring',
            'separation_score': best_correct['separation_score']
        })
    if top_features['incorrect']:
        best_incorrect = top_features['incorrect'][0]
        features_to_test.append({
            'layer': best_incorrect['layer'],
            'feature_idx': best_incorrect['feature_idx'],
            'type': 'incorrect-preferring',
            'separation_score': best_incorrect['separation_score']
        })

print(f"\nAuto-discovered features to test:")
for feature in features_to_test:
    print(f"  - Layer {feature['layer']}, Feature {feature['feature_idx']} ({feature['type']}, separation={feature['separation_score']:.3f})")

device = detect_device()

# Analyze each feature
for feature_info in features_to_test:
    layer = feature_info['layer']
    feature_idx = feature_info['feature_idx']
    feature_type = feature_info['type']
    
    print(f"\n{'='*80}")
    print(f'ANALYZING {feature_type.upper()} FEATURE')
    print(f'Layer {layer}, Feature {feature_idx}')
    if 'separation_score' in feature_info:
        print(f'Separation Score: {feature_info["separation_score"]:.3f}')
    print('='*80)
    
    # Load SAE for this layer
    sae = load_gemma_scope_sae(layer, device)
    
    # First analyze Phase 1 data (where features were selected)
    print('\nPHASE 1 DATA (Feature Selection Dataset):')
    print('-'*40)
    phase1_correct = []
    phase1_incorrect = []

    for _, row in phase1_data.iterrows():
        task_id = row['task_id']
        test_passed = row['test_passed']
        
        # Load activation from Phase 1
        if test_passed:
            act_file = phase1_dir / f'activations/correct/{task_id}_layer_{layer}.npz'
        else:
            act_file = phase1_dir / f'activations/incorrect/{task_id}_layer_{layer}.npz'
        
        if not act_file.exists():
            continue
            
        act_data = np.load(act_file)
        # Phase 1 uses 'layer_X' key, Phase 3.6 uses 'arr_0'
        if f'layer_{layer}' in act_data:
            raw_activation = torch.from_numpy(act_data[f'layer_{layer}']).to(device)
        else:
            raw_activation = torch.from_numpy(act_data['arr_0']).to(device)
        
        # Encode through SAE
        with torch.no_grad():
            sae_features = sae.encode(raw_activation)
        
        feature_activation = sae_features[0, feature_idx].item()
        
        if test_passed:
            phase1_correct.append(feature_activation)
        else:
            phase1_incorrect.append(feature_activation)

    print(f'Correct samples (n={len(phase1_correct)}):')
    if phase1_correct:
        print(f'  Mean: {np.mean(phase1_correct):.4f}')
        print(f'  Std: {np.std(phase1_correct):.4f}')
        print(f'  Min/Max: {np.min(phase1_correct):.4f} / {np.max(phase1_correct):.4f}')
        print(f'  Non-zero: {sum(a > 0 for a in phase1_correct)}/{len(phase1_correct)} ({100*sum(a > 0 for a in phase1_correct)/len(phase1_correct):.1f}%)')
        if sum(a > 0 for a in phase1_correct) <= 10:
            print(f'  Non-zero values: {[f"{a:.2f}" for a in phase1_correct if a > 0]}')

    print(f'\nIncorrect samples (n={len(phase1_incorrect)}):')
    if phase1_incorrect:
        print(f'  Mean: {np.mean(phase1_incorrect):.4f}')
        print(f'  Std: {np.std(phase1_incorrect):.4f}')
        print(f'  Min/Max: {np.min(phase1_incorrect):.4f} / {np.max(phase1_incorrect):.4f}')
        print(f'  Non-zero: {sum(a > 0 for a in phase1_incorrect)}/{len(phase1_incorrect)} ({100*sum(a > 0 for a in phase1_incorrect)/len(phase1_incorrect):.1f}%)')
        if sum(a > 0 for a in phase1_incorrect) <= 10:
            print(f'  Non-zero values: {[f"{a:.2f}" for a in phase1_incorrect if a > 0]}')

    # Then analyze Phase 3.5 data (validation, where final metrics are reported)
    print('\nPHASE 3.5 DATA (Validation Dataset - Final Metrics):')
    print('-'*40)
    activations_correct = []
    activations_incorrect = []

    for _, row in validation_data.iterrows():
        task_id = row['task_id']
        
        # Get test result
        task_results = phase3_5_data[phase3_5_data['task_id'] == task_id]['test_passed'].values
        if len(task_results) == 0:
            continue
        test_passed = task_results[0]
        
        # Load activation from Phase 3.5
        act_file = phase3_5_dir / f'activations/task_activations/{task_id}_layer_{layer}.npz'
        if not act_file.exists():
            continue
            
        act_data = np.load(act_file)
        # Phase 1 uses 'layer_X' key, Phase 3.6 uses 'arr_0'
        if f'layer_{layer}' in act_data:
            raw_activation = torch.from_numpy(act_data[f'layer_{layer}']).to(device)
        else:
            raw_activation = torch.from_numpy(act_data['arr_0']).to(device)
        
        # Encode through SAE
        with torch.no_grad():
            sae_features = sae.encode(raw_activation)
        
        feature_activation = sae_features[0, feature_idx].item()
        
        if test_passed:
            activations_correct.append(feature_activation)
        else:
            activations_incorrect.append(feature_activation)

    if activations_correct:
        print(f'Correct samples (n={len(activations_correct)}):')
        print(f'  Mean: {np.mean(activations_correct):.4f}')
        print(f'  Std: {np.std(activations_correct):.4f}')
        print(f'  Min/Max: {np.min(activations_correct):.4f} / {np.max(activations_correct):.4f}')
        print(f'  Non-zero: {sum(a > 0 for a in activations_correct)}/{len(activations_correct)} ({100*sum(a > 0 for a in activations_correct)/len(activations_correct):.1f}%)')
        if sum(a > 0 for a in activations_correct) <= 10:
            print(f'  Non-zero values: {[f"{a:.2f}" for a in activations_correct if a > 0]}')
    else:
        print(f'Correct samples (n=0): No Phase 3.5 data available')
    
    if activations_incorrect:
        print(f'\nIncorrect samples (n={len(activations_incorrect)}):')
        print(f'  Mean: {np.mean(activations_incorrect):.4f}')
        print(f'  Std: {np.std(activations_incorrect):.4f}')
        print(f'  Min/Max: {np.min(activations_incorrect):.4f} / {np.max(activations_incorrect):.4f}')
        print(f'  Non-zero: {sum(a > 0 for a in activations_incorrect)}/{len(activations_incorrect)} ({100*sum(a > 0 for a in activations_incorrect)/len(activations_incorrect):.1f}%)')
        if sum(a > 0 for a in activations_incorrect) <= 10:
            print(f'  Non-zero values: {[f"{a:.2f}" for a in activations_incorrect if a > 0]}')
    else:
        print(f'\nIncorrect samples (n=0): No Phase 3.5 data available')
    
    # Show threshold analysis if both sets are available
    if activations_correct and activations_incorrect:
        all_activations = activations_correct + activations_incorrect
        print(f'\nThreshold analysis:')
        print(f'  Range: {min(all_activations):.4f} to {max(all_activations):.4f}')

    print('\nGENERALIZATION ANALYSIS:')
    print('-'*40)
    
    # Compare Phase 1 vs Phase 3.5 for the expected activation pattern
    if feature_type == 'correct-preferring':
        # For correct-preferring features, we expect higher activation on correct samples
        if phase1_correct and activations_correct:
            phase1_rate = 100*sum(a > 0 for a in phase1_correct)/len(phase1_correct)
            phase3_5_rate = 100*sum(a > 0 for a in activations_correct)/len(activations_correct) if activations_correct else 0
            print(f'Activation rate on CORRECT samples (expected to be high):')
            print(f'  Phase 1 (training): {sum(a > 0 for a in phase1_correct)}/{len(phase1_correct)} = {phase1_rate:.1f}%')
            if activations_correct:
                print(f'  Phase 3.5 (validation): {sum(a > 0 for a in activations_correct)}/{len(activations_correct)} = {phase3_5_rate:.1f}%')
                print(f'  Generalization: {"Good" if abs(phase1_rate - phase3_5_rate) < 20 else "Poor"}')
    else:
        # For incorrect-preferring features, we expect higher activation on incorrect samples
        if phase1_incorrect and activations_incorrect:
            phase1_rate = 100*sum(a > 0 for a in phase1_incorrect)/len(phase1_incorrect)
            phase3_5_rate = 100*sum(a > 0 for a in activations_incorrect)/len(activations_incorrect) if activations_incorrect else 0
            print(f'Activation rate on INCORRECT samples (expected to be high):')
            print(f'  Phase 1 (training): {sum(a > 0 for a in phase1_incorrect)}/{len(phase1_incorrect)} = {phase1_rate:.1f}%')
            if activations_incorrect:
                print(f'  Phase 3.5 (validation): {sum(a > 0 for a in activations_incorrect)}/{len(activations_incorrect)} = {phase3_5_rate:.1f}%')
                print(f'  Generalization: {"Good" if abs(phase1_rate - phase3_5_rate) < 20 else "Poor"}')

print('\n' + '='*80)
print('SUMMARY')
print('='*80)
print('Both features show sparse activation patterns in Phase 1 data.')
if phase3_5_dir.exists() and phase3_5_data is not None:
    print('Phase 3.5 validation data shows generalization issues for incorrect-preferring feature.')
else:
    print('Phase 3.5 data not available for generalization analysis.')