#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

cd /home/kriz.tahimic/sae-code-correctness

echo "=============================="
echo "Deleting stale probe data..."
echo "=============================="
rm -rf data/phase4_5_probe data/phase4_6_probe data/phase4_8_probe

echo "=============================="
echo "Phase 4.5: Coefficient grid search (probe, --parallel 4)"
echo "=============================="
python3 run.py phase 4.5 --direction-source probe_mass_mean --parallel 4

echo "=============================="
echo "Phase 4.6: Golden section refinement (probe, --parallel 4)"
echo "=============================="
python3 run.py phase 4.6 --direction-source probe_mass_mean --parallel 4

echo "=============================="
echo "Phase 4.8: Steering effect analysis (probe, --parallel 4)"
echo "=============================="
python3 run.py phase 4.8 --direction-source probe_mass_mean --parallel 4

echo "=============================="
echo "All probe phases complete!"
echo "=============================="
