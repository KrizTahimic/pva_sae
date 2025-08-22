#!/usr/bin/env python3
"""Test that Phase 2.5 sorting is now deterministic."""

import json
from pathlib import Path
import hashlib

def get_feature_fingerprint(features_list):
    """Create a deterministic fingerprint of a feature list."""
    # Create a string representation that captures order and values
    feature_str = json.dumps([
        (f['feature_idx'], f['layer'], f['separation_score']) 
        for f in features_list
    ], sort_keys=True)
    
    # Return hash for easy comparison
    return hashlib.md5(feature_str.encode()).hexdigest()

def main():
    """Check if Phase 2.5 results are deterministic."""
    
    phase2_5_dir = Path("data/phase2_5")
    top_20_file = phase2_5_dir / "top_20_features.json"
    
    if not top_20_file.exists():
        print("❌ No Phase 2.5 results found. Run Phase 2.5 first.")
        return
    
    with open(top_20_file, 'r') as f:
        data = json.load(f)
    
    print("=" * 60)
    print("PHASE 2.5 DETERMINISM CHECK")
    print("=" * 60)
    
    # Get fingerprints
    correct_fingerprint = get_feature_fingerprint(data['correct'])
    incorrect_fingerprint = get_feature_fingerprint(data['incorrect'])
    
    print(f"\nCorrect features fingerprint:   {correct_fingerprint}")
    print(f"Incorrect features fingerprint: {incorrect_fingerprint}")
    
    # Check for ties and their ordering
    print("\n" + "-" * 40)
    print("CHECKING TIE-BREAKING:")
    print("-" * 40)
    
    for category in ['correct', 'incorrect']:
        features = data[category]
        
        print(f"\n{category.upper()} features with ties:")
        prev_score = None
        ties = []
        
        for i, feat in enumerate(features):
            if prev_score == feat['separation_score']:
                if not ties or ties[-1]['score'] != feat['separation_score']:
                    ties.append({
                        'score': feat['separation_score'],
                        'features': [(features[i-1]['feature_idx'], features[i-1]['layer'])]
                    })
                ties[-1]['features'].append((feat['feature_idx'], feat['layer']))
            prev_score = feat['separation_score']
        
        if ties:
            for tie_group in ties:
                print(f"  Score {tie_group['score']:.10f}:")
                for feat_idx, layer in tie_group['features']:
                    print(f"    - Feature {feat_idx} (layer {layer})")
                # Check if tie-breaking is deterministic (sorted by layer, then feature_idx)
                is_sorted = all(
                    (tie_group['features'][i][0] < tie_group['features'][i+1][0] or 
                     (tie_group['features'][i][0] == tie_group['features'][i+1][0] and 
                      tie_group['features'][i][1] <= tie_group['features'][i+1][1]))
                    for i in range(len(tie_group['features']) - 1)
                )
                if is_sorted:
                    print("    ✓ Deterministic ordering (by layer, then feature_idx)")
                else:
                    print("    ❌ Non-deterministic ordering detected!")
        else:
            print("  No ties found")
    
    print("\n" + "=" * 60)
    print("To verify determinism:")
    print("1. Save these fingerprints")
    print("2. Re-run Phase 2.5")  
    print("3. Run this script again")
    print("4. Compare fingerprints - they should be identical")
    print("=" * 60)
    
    # Save fingerprints for comparison
    fingerprint_file = phase2_5_dir / "determinism_check.json"
    fingerprints = {
        'correct_fingerprint': correct_fingerprint,
        'incorrect_fingerprint': incorrect_fingerprint
    }
    
    if fingerprint_file.exists():
        with open(fingerprint_file, 'r') as f:
            old_fingerprints = json.load(f)
        
        print("\n🔍 Comparing with previous run:")
        if old_fingerprints == fingerprints:
            print("✅ IDENTICAL - Results are deterministic!")
        else:
            print("❌ DIFFERENT - Results changed between runs")
            print(f"   Old correct:   {old_fingerprints.get('correct_fingerprint', 'N/A')}")
            print(f"   New correct:   {correct_fingerprint}")
            print(f"   Old incorrect: {old_fingerprints.get('incorrect_fingerprint', 'N/A')}")
            print(f"   New incorrect: {incorrect_fingerprint}")
    
    with open(fingerprint_file, 'w') as f:
        json.dump(fingerprints, f, indent=2)
    
    print(f"\nFingerprints saved to: {fingerprint_file}")

if __name__ == "__main__":
    main()