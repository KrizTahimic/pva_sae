#!/usr/bin/env python3
"""Verify Phase 5.3 parallel merge selected the best candidate.

Reads existing orthogonalization_results.json files and checks that
incorrect_orthogonalization.candidate matches the highest-rate candidate
in multi_candidate.per_candidate.

NOTE: per_candidate rates in merged JSON are from GPU-0's problem subset
only (not global). A match here is a necessary but not sufficient condition
for correctness. A mismatch is definitive evidence of a bug.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def check(path: Path) -> bool:
    with open(path) as f:
        d = json.load(f)

    ok = True
    for direction in ['incorrect', 'correct']:
        key = f'{direction}_orthogonalization'
        rate_key = 'correction_rate' if direction == 'incorrect' else 'corruption_rate'

        per = d.get('multi_candidate', {}).get(direction, {}).get('per_candidate', {})
        if not per:
            print(f"  [{direction}] No multi_candidate data — single-GPU run, skipping.")
            continue

        best_id = max(per, key=lambda k: per[k]['metrics'].get(rate_key, 0))
        stored_id = d.get('best_selection', {}).get(direction, '?')

        # per_candidate rates are GPU-0 subset only — NOT comparable to global stored_rate
        subset_rate = per[best_id]['metrics'].get(rate_key, 0)
        global_rate = d.get(key, {}).get('metrics', {}).get(rate_key, 0)
        match = stored_id == best_id
        flag = "✓" if match else "✗ MISMATCH"
        print(f"  [{direction}] stored={stored_id} (global={global_rate:.1f}%), "
              f"best_in_per_candidate={best_id} (GPU-0 subset={subset_rate:.1f}%)  {flag}")
        if not match:
            ok = False
    return ok


all_ok = True
found = False
for results_file in sorted(DATA_DIR.glob("phase5_3*/orthogonalization_results.json")):
    found = True
    parallel = json.loads(results_file.read_text()).get('parallel_merge', False)
    print(f"\n{results_file.parent.name}  (parallel_merge={parallel})")
    ok = check(results_file)
    all_ok = all_ok and ok

if not found:
    print("No orthogonalization_results.json files found under data/phase5_3*/")

print("\n" + ("All checks passed ✓" if all_ok else "MISMATCHES FOUND — consider rerunning Phase 5.3"))
