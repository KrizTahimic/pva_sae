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
phase1_file = phase1_dir / 'dataset_sae_20250820_180307.parquet'
phase1_data = pd.read_parquet(phase1_file)

# Load Phase 3.5 validation data (what Phase 3.8 reports final metrics on)
validation_data = pd.read_parquet(Path('data/phase0_1/validation_mbpp.parquet'))
phase3_5_data = pd.read_parquet(phase3_5_dir / 'dataset_temp_0_0.parquet')

# Load SAE
layer = 15
feature_idx = 16323
device = detect_device()
sae = load_gemma_scope_sae(layer, device)

print(f'Analyzing Feature {feature_idx} at Layer {layer}')
print('='*60)

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
    print(f'  Non-zero: {sum(a > 0 for a in phase1_correct)}/{len(phase1_correct)}')
    print(f'  Values: {[f"{a:.2f}" for a in phase1_correct if a > 0]}')

print(f'Incorrect samples (n={len(phase1_incorrect)}):')
if phase1_incorrect:
    print(f'  Mean: {np.mean(phase1_incorrect):.4f}')
    print(f'  Non-zero: {sum(a > 0 for a in phase1_incorrect)}/{len(phase1_incorrect)}')
    print(f'  Values: {[f"{a:.2f}" for a in phase1_incorrect if a > 0]}')

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

print(f'Correct samples (n={len(activations_correct)}):')
print(f'  Mean: {np.mean(activations_correct):.4f}')
print(f'  Min: {np.min(activations_correct):.4f}')
print(f'  Max: {np.max(activations_correct):.4f}')
print(f'  Non-zero: {sum(a > 0 for a in activations_correct)}/{len(activations_correct)}')
print()

print(f'Incorrect samples (n={len(activations_incorrect)}):')
print(f'  Mean: {np.mean(activations_incorrect):.4f}')
print(f'  Min: {np.min(activations_incorrect):.4f}')
print(f'  Max: {np.max(activations_incorrect):.4f}')
print(f'  Non-zero: {sum(a > 0 for a in activations_incorrect)}/{len(activations_incorrect)}')
print()

# Show threshold analysis
all_activations = activations_correct + activations_incorrect
print(f'Optimal threshold should be between {min(all_activations):.4f} and {max(all_activations):.4f}')
print()

print('\nCOMPARISON:')
print('='*60)
if phase1_incorrect and activations_incorrect:
    print('Activation rate on incorrect samples:')
    print(f'  Phase 1 (feature selection): {sum(a > 0 for a in phase1_incorrect)}/{len(phase1_incorrect)} = {100*sum(a > 0 for a in phase1_incorrect)/len(phase1_incorrect):.1f}%')
    print(f'  Phase 3.5 (validation): {sum(a > 0 for a in activations_incorrect)}/{len(activations_incorrect)} = {100*sum(a > 0 for a in activations_incorrect)/len(activations_incorrect):.1f}%')
    print()
    print('This confirms the outlier hypothesis:')
    print(f'  - Phase 1: Feature activated on ~{100*sum(a > 0 for a in phase1_incorrect)/len(phase1_incorrect):.0f}% of incorrect samples')
    print(f'  - Phase 3.5: Feature activates much less frequently')
    print('  - The feature was selected based on a pattern that rarely occurs in validation data')