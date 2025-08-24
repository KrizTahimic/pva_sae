"""Debug script to identify the Phase 2.10 AUROC issue."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

def analyze_issue():
    """Analyze the discrepancy between Phase 2.10 features and Phase 3.8 evaluation."""
    
    print("=== Debugging Phase 2.10 AUROC Issue ===\n")
    
    # 1. Load Phase 2.10 results
    phase2_10_dir = Path("data/phase2_10")
    if not phase2_10_dir.exists():
        print("ERROR: Phase 2.10 results not found")
        return
    
    with open(phase2_10_dir / "top_20_features.json", 'r') as f:
        top_features = json.load(f)
    
    print("1. Phase 2.10 Top Features:")
    print(f"   Best correct feature: idx={top_features['correct'][0]['feature_idx']}, "
          f"layer={top_features['correct'][0]['layer']}, "
          f"t-stat={top_features['correct'][0]['t_statistic']:.3f}")
    print(f"   Best incorrect feature: idx={top_features['incorrect'][0]['feature_idx']}, "
          f"layer={top_features['incorrect'][0]['layer']}, "
          f"t-stat={top_features['incorrect'][0]['t_statistic']:.3f}")
    
    # 2. Check Phase 1.0 data (source for Phase 2.10)
    phase1_0_dir = Path("data/phase1_0")
    if phase1_0_dir.exists():
        # Load Phase 1.0 dataset
        datasets = list(phase1_0_dir.glob("dataset_*.parquet"))
        if datasets:
            df_phase1 = pd.read_parquet(datasets[0])
            print(f"\n2. Phase 1.0 Data (source for Phase 2.10):")
            print(f"   Total tasks: {len(df_phase1)}")
            print(f"   Correct: {df_phase1['test_passed'].sum()}")
            print(f"   Incorrect: {(~df_phase1['test_passed']).sum()}")
            
            # Get sample task IDs
            sample_correct = df_phase1[df_phase1['test_passed']].iloc[:3]['task_id'].values
            sample_incorrect = df_phase1[~df_phase1['test_passed']].iloc[:3]['task_id'].values
            print(f"   Sample correct tasks: {list(sample_correct)}")
            print(f"   Sample incorrect tasks: {list(sample_incorrect)}")
    
    # 3. Check Phase 3.5 data (used for evaluation)
    phase3_5_dir = Path("data/phase3_5")
    if phase3_5_dir.exists():
        df_phase3_5 = pd.read_parquet(phase3_5_dir / "dataset_temp_0_0.parquet")
        print(f"\n3. Phase 3.5 Data (evaluation data):")
        print(f"   Total tasks: {len(df_phase3_5)}")
        print(f"   Correct: {df_phase3_5['test_passed'].sum()}")
        print(f"   Incorrect: {(~df_phase3_5['test_passed']).sum()}")
        
        # Check if the same tasks have different labels
        if phase1_0_dir.exists() and datasets:
            common_tasks = set(df_phase1['task_id']) & set(df_phase3_5['task_id'])
            if common_tasks:
                print(f"\n4. Label Consistency Check:")
                print(f"   Common tasks between Phase 1.0 and Phase 3.5: {len(common_tasks)}")
                
                mismatches = []
                for task_id in list(common_tasks)[:10]:  # Check first 10
                    label_p1 = df_phase1[df_phase1['task_id'] == task_id]['test_passed'].iloc[0]
                    label_p3_5 = df_phase3_5[df_phase3_5['task_id'] == task_id]['test_passed'].iloc[0]
                    
                    if label_p1 != label_p3_5:
                        mismatches.append({
                            'task_id': task_id,
                            'phase1_0': label_p1,
                            'phase3_5': label_p3_5
                        })
                
                if mismatches:
                    print(f"   CRITICAL: Found {len(mismatches)} label mismatches!")
                    for m in mismatches[:3]:
                        print(f"     Task {m['task_id']}: Phase1.0={m['phase1_0']}, Phase3.5={m['phase3_5']}")
                else:
                    print("   Labels are consistent between phases")
    
    # 5. Check activation files
    print(f"\n5. Activation Files Check:")
    phase1_act_dir = phase1_0_dir / "activations" if phase1_0_dir.exists() else None
    if phase1_act_dir and phase1_act_dir.exists():
        correct_acts = list((phase1_act_dir / "correct").glob("*.npz"))
        incorrect_acts = list((phase1_act_dir / "incorrect").glob("*.npz"))
        print(f"   Phase 1.0 activations: {len(correct_acts)} correct, {len(incorrect_acts)} incorrect")
    
    phase3_5_act_dir = phase3_5_dir / "activations/task_activations" if phase3_5_dir.exists() else None
    if phase3_5_act_dir and phase3_5_act_dir.exists():
        acts = list(phase3_5_act_dir.glob("*.npz"))
        print(f"   Phase 3.5 activations: {len(acts)} total")
    
    # 6. Verify t-statistic calculation
    print(f"\n6. T-Statistic Verification:")
    print("   Testing with synthetic data...")
    
    # Create synthetic data where correct > incorrect
    np.random.seed(42)
    correct_synthetic = np.random.normal(5, 1, 100)  # mean=5
    incorrect_synthetic = np.random.normal(2, 1, 100)  # mean=2
    
    t_stat = stats.ttest_ind(correct_synthetic, incorrect_synthetic, equal_var=False).statistic
    print(f"   Synthetic test (correct mean=5, incorrect mean=2):")
    print(f"     t-statistic = {t_stat:.3f} (should be positive)")
    
    # The logic in Phase 2.10:
    # - Stores t_stat for correct-preferring (should be positive when correct > incorrect)
    # - Stores -t_stat for incorrect-preferring (should be positive when incorrect > correct)
    print(f"     For correct-preferring: {t_stat:.3f}")
    print(f"     For incorrect-preferring: {-t_stat:.3f}")
    
    print("\n=== Analysis Summary ===")
    print("The issue appears to be that Phase 2.10 computes t-statistics on Phase 1.0 data,")
    print("but Phase 3.8 evaluates on Phase 3.5/3.6 data which may have different labels")
    print("or different activation patterns due to being separate generation runs.")
    print("\nPossible solutions:")
    print("1. Compute t-statistics on the same data used for evaluation (Phase 3.5/3.6)")
    print("2. Use Phase 1.0 data for both feature selection AND evaluation")
    print("3. Ensure consistent task splits and labels across phases")

if __name__ == "__main__":
    analyze_issue()