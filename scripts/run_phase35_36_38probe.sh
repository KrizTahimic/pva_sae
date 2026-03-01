#!/bin/bash
# Rerun Phase 3.5 (temp=0 only) + Phase 3.6 + Phase 3.8 probe
# Extracts SAE + probe layers so Phase 3.8 probe has all top-N candidates.
# Usage: screen -dmS phase35_36_38probe bash scripts/run_phase35_36_38probe.sh

set -e  # Exit on error

LOG_FILE="scripts/phase35_36_38probe.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Phase 3.5 / 3.6 / 3.8 Probe Pipeline"
echo "Started: $(date)"
echo "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

echo ""
echo "=========================================="
echo "Phase 3.5 (temp=0 only, SAE+probe layers)"
echo "=========================================="
python3 run.py phase 3.5 --parallel 4

echo ""
echo "=========================================="
echo "Phase 3.6 (tuning split, SAE+probe layers)"
echo "=========================================="
python3 run.py phase 3.6 --parallel 4

echo ""
echo "=========================================="
echo "Phase 3.8 - Probe (top-N layer selection)"
echo "=========================================="
python3 run.py phase 3.8 --direction-source probe_logreg

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "Finished: $(date)"
echo "=========================================="
