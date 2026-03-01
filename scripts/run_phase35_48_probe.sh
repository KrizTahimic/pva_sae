#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

cd /home/kriz.tahimic/sae-code-correctness

echo "=============================="
echo "Phase 3.5: Temperature sweep (parallel 4)"
echo "=============================="
python3 run.py phase 3.5 --parallel 4

echo "=============================="
echo "Phase 4.8: Steering effect analysis (probe, --parallel 4)"
echo "=============================="
python3 run.py phase 4.8 --direction-source probe_mass_mean --parallel 4

echo "=============================="
echo "Done!"
echo "=============================="
