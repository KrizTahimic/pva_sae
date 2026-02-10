#!/bin/bash
# Run Phase 4.5 and 4.6 for all three models with parallel 4
# Tests top-5 SAE steering latents per model
#
# Prerequisites (for each model):
# - Phase 2.5: data/phase2_5{suffix}/top_20_latents.json
# - Phase 3.6: data/phase3_6{suffix}/dataset_hyperparams_temp_0_0.parquet
#
# Usage:
#   ./scripts/run_phase4_all_models.sh           # Full run
#   ./scripts/run_phase4_all_models.sh --test    # Test with subset (--start 0 --end 10)

set -e  # Exit on error

# Parse arguments
TEST_MODE=""
if [[ "$1" == "--test" ]]; then
    TEST_MODE="--start 0 --end 10"
    echo "Running in TEST MODE with $TEST_MODE"
fi

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

MODELS=(
    "google/gemma-2-2b"
    "google/gemma-2-9b"
    "meta-llama/Llama-3.1-8B"
)

for model in "${MODELS[@]}"; do
    echo "=============================================="
    echo "Phase 4.5: $model"
    echo "=============================================="
    python3 run.py phase 4.5 --model "$model" --parallel 4 $TEST_MODE

    echo "=============================================="
    echo "Phase 4.6: $model"
    echo "=============================================="
    python3 run.py phase 4.6 --model "$model" --parallel 4 $TEST_MODE
done

echo ""
echo "=============================================="
echo "All models complete!"
echo "=============================================="
echo "Outputs:"
echo "  Phase 4.5: data/phase4_5/, data/phase4_5_gemma9b/, data/phase4_5_llama/"
echo "  Phase 4.6: data/phase4_6/, data/phase4_6_gemma9b/, data/phase4_6_llama/"
