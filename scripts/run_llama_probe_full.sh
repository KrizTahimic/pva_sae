#!/bin/bash
# LLAMA-3.1-8B full probe pipeline: 2.6 → 3.6 → 3.8 probe → 4.5 probe → 4.6 probe → 4.8 probe → 5.3 probe → 6.3 probe
#
# Why each phase:
#   2.6  — regenerate with new code to get top_n_probe_directions.json
#   3.6  — regenerate to capture missing probe layers 18/19/20 (union of SAE + probe layers)
#   3.8  — first probe AUROC/F1 for LLAMA
#   4.5  — probe coefficient grid search
#   4.6  — probe golden section refinement
#   4.8  — probe steering effect analysis
#   5.3  — probe orthogonalization
#   6.3  — probe attention analysis
#
# Usage:
#   screen -dmS llama_probe bash scripts/run_llama_probe_full.sh

set -e

LOG_FILE="scripts/run_llama_probe_full.log"
exec > >(tee -a "$LOG_FILE") 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

cd /home/kriz.tahimic/sae-code-correctness

MODEL="meta-llama/Llama-3.1-8B"

echo "============================================================"
echo "LLAMA-3.1-8B Full Probe Pipeline"
echo "Started: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "Model: $MODEL"
echo "============================================================"

echo ""
echo "=============================="
echo "Phase 2.6: Probe training (LLAMA)"
echo "=============================="
python3 run.py phase 2.6 --model "$MODEL"

echo ""
echo "=============================="
echo "Phase 3.6: Hyperparameter baseline — activations (LLAMA, --parallel 4)"
echo "=============================="
python3 run.py phase 3.6 --model "$MODEL" --parallel 4

echo ""
echo "=============================="
echo "Phase 3.8: AUROC/F1 (probe_logreg, LLAMA)"
echo "=============================="
python3 run.py phase 3.8 --direction-source probe_logreg --model "$MODEL"

echo ""
echo "=============================="
echo "Phase 4.5: Coefficient grid search (probe_mass_mean, LLAMA, --parallel 4)"
echo "=============================="
python3 run.py phase 4.5 --direction-source probe_mass_mean --model "$MODEL" --parallel 4

echo ""
echo "=============================="
echo "Phase 4.6: Golden section refinement (probe_mass_mean, LLAMA, --parallel 4)"
echo "=============================="
python3 run.py phase 4.6 --direction-source probe_mass_mean --model "$MODEL" --parallel 4

echo ""
echo "=============================="
echo "Phase 4.8: Steering effect analysis (probe_mass_mean, LLAMA, --parallel 4)"
echo "=============================="
python3 run.py phase 4.8 --direction-source probe_mass_mean --model "$MODEL" --parallel 4

echo ""
echo "=============================="
echo "Phase 5.3: Orthogonalization (probe_mass_mean, LLAMA, --parallel 4)"
echo "=============================="
python3 run.py phase 5.3 --direction-source probe_mass_mean --model "$MODEL" --parallel 4

echo ""
echo "=============================="
echo "Phase 6.3: Attention analysis (probe_mass_mean, LLAMA)"
echo "=============================="
python3 run.py phase 6.3 --direction-source probe_mass_mean --model "$MODEL"

echo ""
echo "============================================================"
echo "LLAMA probe pipeline complete! $(date)"
echo "============================================================"
