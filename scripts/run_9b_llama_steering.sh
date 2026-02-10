#!/bin/bash
# Run Phase 4.5/4.6/4.8 for 9B and LLAMA models
# Usage: screen -dmS multi_model bash scripts/run_9b_llama_steering.sh

set -e  # Exit on error

LOG_FILE="phase_9b_llama_steering.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Multi-Model Steering Pipeline"
echo "Started: $(date)"
echo "=========================================="

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

# ==========================================
# GEMMA 9B
# ==========================================
echo ""
echo "=========================================="
echo "GEMMA 9B - Phase 4.6 (corruption-only)"
echo "=========================================="
python3 run.py phase 4.6 --model google/gemma-2-9b --corruption-only --parallel 4

echo ""
echo "=========================================="
echo "GEMMA 9B - Phase 4.8 (full)"
echo "=========================================="
python3 run.py phase 4.8 --model google/gemma-2-9b --parallel 4

# ==========================================
# LLAMA 8B
# ==========================================
echo ""
echo "=========================================="
echo "LLAMA 8B - Phase 4.5 (full)"
echo "=========================================="
python3 run.py phase 4.5 --model meta-llama/Llama-3.1-8B --parallel 4

echo ""
echo "=========================================="
echo "LLAMA 8B - Phase 4.6 (full)"
echo "=========================================="
python3 run.py phase 4.6 --model meta-llama/Llama-3.1-8B --parallel 4

echo ""
echo "=========================================="
echo "LLAMA 8B - Phase 4.8 (full)"
echo "=========================================="
python3 run.py phase 4.8 --model meta-llama/Llama-3.1-8B --parallel 4

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "Finished: $(date)"
echo "=========================================="
