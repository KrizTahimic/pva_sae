#!/usr/bin/env python3
"""Debug script to identify sources of variability in Phase 1 and Phase 2.5 results."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd

def analyze_activation_files(phase1_dir: Path) -> Dict:
    """Analyze activation files from Phase 1 output."""
    
    activation_dir = phase1_dir / "activations"
    if not activation_dir.exists():
        return {"error": f"Activation directory not found: {activation_dir}"}
    
    correct_dir = activation_dir / "correct"
    incorrect_dir = activation_dir / "incorrect"
    
    # Get all activation files
    correct_files = sorted(list(correct_dir.glob("*.npz")))
    incorrect_files = sorted(list(incorrect_dir.glob("*.npz")))
    
    # Extract task IDs
    correct_task_ids = set()
    incorrect_task_ids = set()
    
    for f in correct_files:
        parts = f.stem.split('_layer_')
        if len(parts) == 2:
            correct_task_ids.add(parts[0])
    
    for f in incorrect_files:
        parts = f.stem.split('_layer_')
        if len(parts) == 2:
            incorrect_task_ids.add(parts[0])
    
    # Check for any overlap (shouldn't be any)
    overlap = correct_task_ids & incorrect_task_ids
    
    # Sample some activations to check values
    sample_activations = {}
    if correct_files:
        # Load first few correct activations
        for f in correct_files[:3]:
            data = np.load(f)
            act = data[data.files[0]]
            sample_activations[f.name] = {
                "shape": act.shape,
                "mean": float(np.mean(act)),
                "std": float(np.std(act)),
                "min": float(np.min(act)),
                "max": float(np.max(act)),
                "nonzero_count": int(np.count_nonzero(act))
            }
    
    return {
        "n_correct_files": len(correct_files),
        "n_incorrect_files": len(incorrect_files),
        "n_correct_tasks": len(correct_task_ids),
        "n_incorrect_tasks": len(incorrect_task_ids),
        "task_overlap": list(overlap),
        "sample_activations": sample_activations
    }

def analyze_phase2_5_results(phase2_5_dir: Path) -> Dict:
    """Analyze Phase 2.5 output for variability."""
    
    top_20_file = phase2_5_dir / "top_20_features.json"
    if not top_20_file.exists():
        return {"error": f"Top 20 features file not found: {top_20_file}"}
    
    with open(top_20_file, 'r') as f:
        top_20 = json.load(f)
    
    # Check for features with very close separation scores
    def find_close_scores(features: List[Dict], threshold: float = 1e-6) -> List:
        close_pairs = []
        for i in range(len(features) - 1):
            diff = abs(features[i]['separation_score'] - features[i+1]['separation_score'])
            if diff < threshold:
                close_pairs.append({
                    'indices': (i, i+1),
                    'features': (features[i]['feature_idx'], features[i+1]['feature_idx']),
                    'scores': (features[i]['separation_score'], features[i+1]['separation_score']),
                    'difference': diff
                })
        return close_pairs
    
    correct_close = find_close_scores(top_20['correct'])
    incorrect_close = find_close_scores(top_20['incorrect'])
    
    # Check for duplicate scores
    correct_scores = [f['separation_score'] for f in top_20['correct']]
    incorrect_scores = [f['separation_score'] for f in top_20['incorrect']]
    
    correct_duplicates = len(correct_scores) - len(set(correct_scores))
    incorrect_duplicates = len(incorrect_scores) - len(set(incorrect_scores))
    
    # Analyze score distribution
    def analyze_scores(scores: List[float]) -> Dict:
        return {
            'min': min(scores),
            'max': max(scores),
            'mean': np.mean(scores),
            'std': np.std(scores),
            'unique_count': len(set(scores))
        }
    
    return {
        'correct': {
            'n_features': len(top_20['correct']),
            'close_scores': correct_close,
            'duplicate_scores': correct_duplicates,
            'score_stats': analyze_scores(correct_scores),
            'top_3_features': [(f['feature_idx'], f['separation_score']) for f in top_20['correct'][:3]]
        },
        'incorrect': {
            'n_features': len(top_20['incorrect']),
            'close_scores': incorrect_close,
            'duplicate_scores': incorrect_duplicates,
            'score_stats': analyze_scores(incorrect_scores),
            'top_3_features': [(f['feature_idx'], f['separation_score']) for f in top_20['incorrect'][:3]]
        }
    }

def compare_multiple_runs(run_dirs: List[Path]) -> Dict:
    """Compare results across multiple Phase 2.5 runs."""
    
    all_results = []
    for run_dir in run_dirs:
        top_20_file = run_dir / "top_20_features.json"
        if top_20_file.exists():
            with open(top_20_file, 'r') as f:
                data = json.load(f)
                all_results.append({
                    'dir': str(run_dir),
                    'data': data
                })
    
    if len(all_results) < 2:
        return {"error": "Need at least 2 runs to compare"}
    
    # Compare top features across runs
    comparison = {
        'correct': {
            'feature_consistency': [],
            'score_variability': []
        },
        'incorrect': {
            'feature_consistency': [],
            'score_variability': []
        }
    }
    
    for category in ['correct', 'incorrect']:
        # Get top 5 features from each run
        all_top_features = []
        for result in all_results:
            top_5 = [(f['feature_idx'], f['layer'], f['separation_score']) 
                     for f in result['data'][category][:5]]
            all_top_features.append(top_5)
        
        # Check consistency
        for i in range(min(5, len(all_top_features[0]))):
            features_at_position = [(run[i][0], run[i][1]) for run in all_top_features]
            scores_at_position = [run[i][2] for run in all_top_features]
            
            comparison[category]['feature_consistency'].append({
                'position': i,
                'features': features_at_position,
                'all_same': len(set(features_at_position)) == 1
            })
            
            comparison[category]['score_variability'].append({
                'position': i,
                'scores': scores_at_position,
                'std': np.std(scores_at_position)
            })
    
    return comparison

def check_file_system_ordering():
    """Check if file system ordering might affect results."""
    
    import tempfile
    import os
    
    # Create temporary directory with test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create files with specific names
        test_files = [
            "Mbpp_10_layer_8.npz",
            "Mbpp_2_layer_8.npz", 
            "Mbpp_100_layer_8.npz",
            "Mbpp_20_layer_8.npz",
            "Mbpp_3_layer_8.npz"
        ]
        
        for fname in test_files:
            (tmppath / fname).touch()
        
        # Test different methods of listing
        glob_order = [f.name for f in tmppath.glob("*.npz")]
        sorted_glob = [f.name for f in sorted(tmppath.glob("*.npz"))]
        listdir_order = os.listdir(tmppath)
        sorted_listdir = sorted(os.listdir(tmppath))
        
        return {
            'glob_order': glob_order,
            'sorted_glob': sorted_glob,
            'listdir_order': listdir_order,
            'sorted_listdir': sorted_listdir,
            'glob_consistent': glob_order == sorted_glob,
            'listdir_consistent': listdir_order == sorted_listdir
        }

def main():
    """Run diagnostic analysis."""
    
    print("=" * 60)
    print("PHASE VARIABILITY DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    
    # Check Phase 1 output
    phase1_dir = Path("data/phase1_0")
    if phase1_dir.exists():
        print("\n1. PHASE 1 ACTIVATION ANALYSIS:")
        print("-" * 40)
        phase1_analysis = analyze_activation_files(phase1_dir)
        print(json.dumps(phase1_analysis, indent=2))
    
    # Check Phase 2.5 output
    phase2_5_dir = Path("data/phase2_5")
    if phase2_5_dir.exists():
        print("\n2. PHASE 2.5 RESULTS ANALYSIS:")
        print("-" * 40)
        phase2_5_analysis = analyze_phase2_5_results(phase2_5_dir)
        print(json.dumps(phase2_5_analysis, indent=2))
    
    # Check file system ordering
    print("\n3. FILE SYSTEM ORDERING TEST:")
    print("-" * 40)
    ordering_test = check_file_system_ordering()
    print(json.dumps(ordering_test, indent=2))
    
    # Key insights
    print("\n4. KEY INSIGHTS:")
    print("-" * 40)
    
    if phase2_5_dir.exists() and 'correct' in phase2_5_analysis:
        # Check for score ties
        correct_close = phase2_5_analysis['correct']['close_scores']
        incorrect_close = phase2_5_analysis['incorrect']['close_scores']
        
        if correct_close:
            print(f"⚠️  Found {len(correct_close)} pairs of correct features with very close scores")
            print(f"   This could cause ordering instability")
            
        if incorrect_close:
            print(f"⚠️  Found {len(incorrect_close)} pairs of incorrect features with very close scores")
            print(f"   This could cause ordering instability")
        
        # Check for duplicates
        if phase2_5_analysis['correct']['duplicate_scores'] > 0:
            print(f"⚠️  Found {phase2_5_analysis['correct']['duplicate_scores']} duplicate scores in correct features")
            print(f"   When scores are identical, feature ordering may vary")
        
        if phase2_5_analysis['incorrect']['duplicate_scores'] > 0:
            print(f"⚠️  Found {phase2_5_analysis['incorrect']['duplicate_scores']} duplicate scores in incorrect features")
    
    # File system ordering
    if not ordering_test['glob_consistent']:
        print("⚠️  glob() returns files in non-deterministic order")
        print("   This could affect which activations are processed")

if __name__ == "__main__":
    main()