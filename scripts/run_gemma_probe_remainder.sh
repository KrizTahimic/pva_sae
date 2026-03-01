#!/bin/bash
# Gemma-2B probe remainder: Phase 3.6 → 3.8 probe → 5.3 probe → 6.3 probe
# Prereqs: phase4_5_probe, phase4_6_probe, phase4_8_probe must be complete.
# Phase 3.6 regenerates activations to capture probe layers (union of SAE + probe layers).
#
# Usage:
#   screen -dmS gemma_probe_remainder bash scripts/run_gemma_probe_remainder.sh

set -e

LOG_FILE="scripts/run_gemma_probe_remainder.log"
exec > >(tee -a "$LOG_FILE") 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

cd /home/kriz.tahimic/sae-code-correctness

echo "============================================================"
echo "Gemma-2B Probe Remainder"
echo "Started: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "============================================================"

echo ""
echo "=============================="
echo "Deleting stale Phase 3.5 data (activations empty from checkpoint-skip rerun)..."
echo "=============================="
rm -rf data/phase3_5

echo ""
echo "=============================="
echo "Phase 3.5: Temperature robustness — regenerate with activations (--parallel 4)"
echo "=============================="
python3 run.py phase 3.5 --parallel 4

echo ""
echo "=============================="
echo "Phase 3.6: Hyperparameter baseline — activations (--parallel 4)"
echo "=============================="
python3 run.py phase 3.6 --parallel 4

echo ""
echo "=============================="
echo "Phase 3.8: AUROC/F1 (probe_logreg)"
echo "=============================="
python3 run.py phase 3.8 --direction-source probe_logreg

echo ""
echo "=============================="
echo "Phase 5.3: Orthogonalization (probe_mass_mean, --parallel 4)"
echo "=============================="
python3 run.py phase 5.3 --direction-source probe_mass_mean --parallel 4

echo ""
echo "=============================="
echo "Phase 6.3: Attention analysis (probe_mass_mean)"
echo "=============================="
python3 run.py phase 6.3 --direction-source probe_mass_mean

echo ""
echo "============================================================"
echo "Gemma probe remainder complete! $(date)"
echo "============================================================"
