#!/usr/bin/env python3
"""
Extract layer 31 activations for all Gemma 9B Phase 7.3 tasks.

Phase 7.3 ran before Phase 3.8 probe existed, so layer 31 was never captured.
This script adds the missing layer 31 activation files using a forward pass
(no generation needed — activations are at last prompt token, before decoding).

Usage:
    python3 scripts/extract_layer31_gemma9b.py
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
from tqdm import tqdm

from common.config import Config
from common.model_loader import load_model_and_tokenizer
from common.activation_hooks import ActivationExtractor
from common.tensor_utils import save_activation
from common.utils import detect_device

TARGET_LAYER = 31
MODEL_NAME = "google/gemma-2-9b-it"
PARQUET = project_root / "data/phase7_3_gemma9b/dataset_instruct_temp_0_0.parquet"
ACT_DIR = project_root / "data/phase7_3_gemma9b/activations/task_activations"


def main():
    config = Config()

    device = detect_device()
    print(f"Loading {MODEL_NAME} on {device}...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=device)
    model.eval()

    df = pd.read_parquet(PARQUET)
    print(f"Loaded {len(df)} tasks from Phase 7.3 parquet")

    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting layer 31"):
        task_id = row['task_id']
        out_path = ACT_DIR / f"{task_id}_layer_{TARGET_LAYER}.safetensors"

        if out_path.exists():
            skipped += 1
            continue

        inputs = tokenizer(
            row['prompt'],
            return_tensors="pt",
            truncation=True,
            max_length=config.activation_max_length,
        ).to(device)

        with ActivationExtractor(model, layers=[TARGET_LAYER]) as extractor:
            with torch.no_grad():
                model(**inputs)
            activations = extractor.get_activations()

        if TARGET_LAYER not in activations:
            print(f"WARNING: No activation captured for task {task_id}")
            continue

        save_activation(activations[TARGET_LAYER], out_path)

    print(f"Done. Skipped {skipped} already-existing files.")


if __name__ == "__main__":
    main()
