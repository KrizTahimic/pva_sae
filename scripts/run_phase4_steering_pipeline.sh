#!/bin/bash
# Run Phase 4 steering pipeline for Gemma 2B after direction normalization fix
#
# Pipeline: 4.5 -> 4.6 -> 4.8 -> 4.10 -> 4.12
#
# This re-runs steering experiments with normalized directions to fix the bug
# where Phase 4.12 (zero-disc control) showed higher effect than Phase 4.8.
#
# Config settings (common/config.py):
# - phase4_n_candidates: 5 (discriminative latents)
# - phase4_12_n_features: 5 (zero-disc latents, matches for fair comparison)
#
# Prerequisites:
# - Phase 2.5: data/phase2_5/top_20_latents.json
# - Phase 3.6: data/phase3_6/dataset_hyperparams_temp_0_0.parquet
#
# Usage:
#   ./scripts/run_phase4_steering_pipeline.sh              # Full run
#   ./scripts/run_phase4_steering_pipeline.sh --test       # Test with subset
#   ./scripts/run_phase4_steering_pipeline.sh --parallel 4 # Use 4 GPUs

set -e  # Exit on error

# Parse arguments
TEST_MODE=""
PARALLEL=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE="--start 0 --end 20"
            echo "Running in TEST MODE with $TEST_MODE"
            shift
            ;;
        --parallel)
            PARALLEL="--parallel $2"
            echo "Using parallel mode with $2 GPUs"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test] [--parallel N]"
            exit 1
            ;;
    esac
done

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

echo "=============================================="
echo "Phase 4 Steering Pipeline - Gemma 2B"
echo "After direction normalization fix"
echo "=============================================="
echo "Testing 5 candidates each for discriminative and zero-disc"
echo ""

# Phase 4.5: Coefficient grid search
echo "=============================================="
echo "Phase 4.5: Coefficient Grid Search"
echo "=============================================="
python3 run.py phase 4.5 $PARALLEL $TEST_MODE

# Phase 4.6: Golden section refinement
echo ""
echo "=============================================="
echo "Phase 4.6: Coefficient Refinement"
echo "=============================================="
python3 run.py phase 4.6 $PARALLEL $TEST_MODE

# Phase 4.8: Steering effect analysis (discriminative)
echo ""
echo "=============================================="
echo "Phase 4.8: Steering Effect Analysis"
echo "=============================================="
python3 run.py phase 4.8 $PARALLEL $TEST_MODE

# Phase 4.10: Zero-discrimination feature selection
echo ""
echo "=============================================="
echo "Phase 4.10: Zero-Disc Feature Selection"
echo "=============================================="
python3 run.py phase 4.10 $TEST_MODE

# Phase 4.12: Zero-discrimination steering (control)
echo ""
echo "=============================================="
echo "Phase 4.12: Zero-Disc Steering Control"
echo "=============================================="
python3 run.py phase 4.12 $PARALLEL $TEST_MODE

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  Phase 4.5:  data/phase4_5/"
echo "  Phase 4.6:  data/phase4_6/"
echo "  Phase 4.8:  data/phase4_8/"
echo "  Phase 4.10: data/phase4_10/"
echo "  Phase 4.12: data/phase4_12/"
echo ""
echo "Expected result after fix:"
echo "  Phase 4.8  (discriminative): Higher correction/corruption rates"
echo "  Phase 4.12 (zero-disc):      ~0% effect (null baseline)"
