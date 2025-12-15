"""
Run Linear Probe Sanity Check on All Models in Parallel

Each model runs on a separate GPU.

Usage:
    python run_all_models_parallel.py
    python run_all_models_parallel.py --layers 15 19 23  # specific layers
"""

import sys
from pathlib import Path
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_model_on_gpu(model: str, gpu_id: int, layers: list[int] = None):
    """Run sanity check for one model on specified GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    script_path = Path(__file__).parent / "run_sanity_check.py"

    if layers:
        # Run specific layers
        results = []
        for layer in layers:
            cmd = [
                sys.executable, str(script_path),
                "--model", model,
                "--layer", str(layer)
            ]
            print(f"[GPU {gpu_id}] Running {model} layer {layer}...")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            results.append((layer, result.stdout, result.stderr))
        return model, results
    else:
        # Run all layers
        cmd = [
            sys.executable, str(script_path),
            "--model", model,
            "--all-layers"
        ]
        print(f"[GPU {gpu_id}] Running {model} all layers...")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        return model, result.stdout, result.stderr


def check_phase1_status():
    """Check which models have Phase 1 data ready."""
    import pandas as pd

    models = {
        "gemma2b": "phase1_0",
        "gemma2b_it": "phase1_0_it",
        "gemma9b": "phase1_0_gemma9b",
        "llama": "phase1_0_llama",
    }

    ready = {}
    for model, folder in models.items():
        path = project_root / "data" / folder
        if path.exists():
            parquets = list(path.glob("*.parquet"))
            if parquets:
                df = pd.read_parquet(parquets[0])
                n_samples = len(df)
                ready[model] = {
                    "samples": n_samples,
                    "ready": n_samples >= 200,  # At least 200 samples for meaningful comparison
                    "correct": (df['baseline_passed'] == True).sum(),
                    "incorrect": (df['baseline_passed'] == False).sum(),
                }
    return ready


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[19],
                        help="Layers to test (default: 19 - middle layer)")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["gemma2b", "gemma9b", "llama"],
                        help="Models to test")
    args = parser.parse_args()

    # Check which models have data
    print("="*60)
    print("Checking Phase 1 Data Status")
    print("="*60)

    status = check_phase1_status()
    for model, info in status.items():
        status_str = "✓ READY" if info["ready"] else "✗ NOT READY"
        print(f"  {model:12} | {info['samples']:4} samples | "
              f"{info['correct']:3}C / {info['incorrect']:3}I | {status_str}")

    # Filter to ready models
    models_to_run = [m for m in args.models if m in status and status[m]["ready"]]

    if not models_to_run:
        print("\nNo models have enough data yet. Run Phase 1 first.")
        print("Minimum 200 samples needed for meaningful comparison.")
        return

    print(f"\nRunning comparison on: {models_to_run}")
    print(f"Layers: {args.layers}")
    print("="*60)

    # Assign GPUs (round-robin)
    gpu_assignments = {model: i % 4 for i, model in enumerate(models_to_run)}

    # Run in parallel
    with ProcessPoolExecutor(max_workers=len(models_to_run)) as executor:
        futures = {
            executor.submit(run_model_on_gpu, model, gpu, args.layers): model
            for model, gpu in gpu_assignments.items()
        }

        results = {}
        for future in as_completed(futures):
            model = futures[future]
            try:
                result = future.result()
                results[model] = result
                print(f"\n{'='*60}")
                print(f"COMPLETED: {model}")
                print("="*60)
                if isinstance(result[1], list):
                    for layer, stdout, stderr in result[1]:
                        print(f"\n--- Layer {layer} ---")
                        print(stdout)
                        if stderr:
                            print(f"STDERR: {stderr}")
                else:
                    print(result[1])
                    if result[2]:
                        print(f"STDERR: {result[2]}")
            except Exception as e:
                print(f"ERROR running {model}: {e}")

    print("\n" + "="*60)
    print("ALL MODELS COMPLETED")
    print("="*60)

    # Save aggregated results
    from datetime import datetime
    output_dir = project_root / "experiments" / "linear_probe_sanity_check" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "layers": args.layers,
        "models": {}
    }

    # Parse results from stdout (simple extraction)
    for model, result_data in results.items():
        summary["models"][model] = {"status": "completed"}

    output_file = output_dir / f"parallel_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        import json
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {output_file}")
    print(f"Individual results in: {output_dir}/")


if __name__ == "__main__":
    main()
