"""
Toy test: does temp=0 generation produce identical outputs across calls?

Test 1 (within-process): Generate the same 5 problems twice consecutively
in the same Python session. They must be identical if generation is
deterministic.

Test 2 (cross-process): Run Phase 3.5 on --start 0 --end 5 twice (with
checkpoints deleted between runs) and compare baseline_passed columns.
Commands are printed at the end.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd

from common.config import Config
from common.initialization import setup_deterministic_generation
from common.model_loader import load_model_and_tokenizer


def generate_once(model, tokenizer, prompt: str, config: Config) -> str:
    setup_deterministic_generation(seed=42)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            temperature=0.0,
            do_sample=False,
            max_new_tokens=config.model_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def main() -> None:
    config = Config()
    setup_deterministic_generation(seed=42)

    model, tokenizer = load_model_and_tokenizer(
        config.model_name, use_eager_attention=True
    )
    model.eval()

    parquet_path = Path("data/phase3_5/dataset_temp_0_0.parquet")
    if not parquet_path.exists():
        print(f"ERROR: {parquet_path} not found. Run Phase 3.5 first.")
        sys.exit(1)

    df = pd.read_parquet(parquet_path).head(5)

    print("=" * 60)
    print("Test 1: within-process determinism (same session, 2 calls)")
    print("=" * 60)

    results = []
    for i, row in df.iterrows():
        prompt = row["prompt"]
        out1 = generate_once(model, tokenizer, prompt, config)
        out2 = generate_once(model, tokenizer, prompt, config)
        match = out1 == out2
        results.append((i, match, out1, out2))
        status = "IDENTICAL" if match else "*** DIFFERENT ***"
        print(f"Problem {i}: {status}")
        if not match:
            print(f"  Run1: {out1[:120]!r}")
            print(f"  Run2: {out2[:120]!r}")

    n_diff = sum(1 for _, m, _, _ in results if not m)
    print(f"\nSummary: {n_diff}/5 problems differ within same process")
    if n_diff == 0:
        print("=> Within-process: DETERMINISTIC. Run Test 2 to check cross-process.")
    else:
        print("=> Non-determinism exists WITHIN a single process (warn_only=True is allowing it).")

    print()
    print("=" * 60)
    print("Test 2: cross-process comparison commands")
    print("=" * 60)
    print("""
# Run 1
rm -rf data/phase3_5_det_test
mkdir -p data/phase3_5_det_test
python3 run.py phase 3.5 --start 0 --end 5 2>&1 | tee /tmp/det_run1.log
cp data/phase3_5/dataset_temp_0_0.parquet data/phase3_5_det_test/run1.parquet

# Run 2 (delete checkpoints, re-run)
find data/phase3_5 -name "*.parquet" -delete
find data/phase3_5 -name "checkpoint*" -delete
python3 run.py phase 3.5 --start 0 --end 5 2>&1 | tee /tmp/det_run2.log
cp data/phase3_5/dataset_temp_0_0.parquet data/phase3_5_det_test/run2.parquet

# Compare baseline_passed
python3 -c "
import pandas as pd
r1 = pd.read_parquet('data/phase3_5_det_test/run1.parquet')
r2 = pd.read_parquet('data/phase3_5_det_test/run2.parquet')
diff = r1['baseline_passed'] != r2['baseline_passed']
print(f'Differing baseline_passed: {diff.sum()}/5')
print(r1[diff][['task_id','baseline_passed']])
print(r2[diff][['task_id','baseline_passed']])
"
""")


if __name__ == "__main__":
    main()
