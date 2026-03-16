"""
Test 2: cross-process determinism.

Generates the same 5 prompts fresh in this new process and compares
the raw generated text against what is stored in the existing
data/phase3_5/dataset_temp_0_0.parquet (Run 1).

If any outputs differ, the CUDA workspace initialises differently across
processes and CUBLAS_WORKSPACE_CONFIG must be set to fix it.
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

    parquet_path = Path("data/phase3_5/dataset_temp_0_0.parquet")
    if not parquet_path.exists():
        print(f"ERROR: {parquet_path} not found.")
        sys.exit(1)

    run1 = pd.read_parquet(parquet_path).head(5)

    model, tokenizer = load_model_and_tokenizer(
        config.model_name, use_eager_attention=True
    )
    model.eval()

    print("=" * 60)
    print("Test 2: cross-process determinism")
    print("Run 1 = existing phase3_5 parquet  |  Run 2 = fresh generation")
    print("=" * 60)

    results = []
    for i, row in run1.iterrows():
        prompt = row["prompt"]
        run1_text = row["generated_code"]
        run2_text = generate_once(model, tokenizer, prompt, config)
        match = run1_text == run2_text
        results.append((i, match, run1_text, run2_text))
        status = "IDENTICAL" if match else "*** DIFFERENT ***"
        print(f"Problem {i} (task_id={row.get('task_id', '?')}): {status}")
        if not match:
            # Find first differing character
            min_len = min(len(run1_text), len(run2_text))
            diff_pos = next(
                (j for j in range(min_len) if run1_text[j] != run2_text[j]),
                min_len,
            )
            print(f"  First diff at char {diff_pos}")
            print(f"  Run1: {run1_text[:120]!r}")
            print(f"  Run2: {run2_text[:120]!r}")

    n_diff = sum(1 for _, m, _, _ in results if not m)
    print(f"\nSummary: {n_diff}/5 problems differ across processes")
    if n_diff == 0:
        print("=> Cross-process: DETERMINISTIC.")
        print("   The Feb 9→Feb 27 discrepancy has another cause.")
    else:
        print("=> NON-DETERMINISM CONFIRMED across process boundaries.")
        print("   Fix: add CUBLAS_WORKSPACE_CONFIG=:4096:8 to the environment.")
        print("   Add to common/initialization.py::setup_deterministic_generation():")
        print("     import os")
        print("     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'")


if __name__ == "__main__":
    main()
