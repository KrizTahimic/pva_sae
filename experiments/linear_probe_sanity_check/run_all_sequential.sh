#!/bin/bash
# Run linear probe sanity check for all models, all layers
# Sequential execution - takes ~36 minutes total

set -e

cd "$(dirname "$0")/../.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

echo "=============================================="
echo "Linear Probe vs SAE Sanity Check"
echo "All models, all layers (10-25)"
echo "=============================================="

# Run each model sequentially
for model in gemma2b gemma9b llama; do
    echo ""
    echo ">>> Starting $model (all layers)..."
    python experiments/linear_probe_sanity_check/run_sanity_check.py \
        --model $model \
        --all-layers
    echo ">>> Completed $model"
done

echo ""
echo "=============================================="
echo "ALL DONE! Results in:"
echo "  experiments/linear_probe_sanity_check/results/"
echo "=============================================="
