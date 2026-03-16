#!/bin/bash
# Re-sync Gemma-2-2B pipeline from Phase 3.5 onward (SAE + Probe).
#
# Context: CUBLAS non-determinism caused ~3% of baseline_passed labels to
# flip across Phase 3.5 runs. The Mar 6 Phase 3.5 data is the validated
# CUBLAS-fixed baseline (0/50 flips confirmed). This script re-runs every
# downstream phase that is stale relative to that baseline.
#
# Preserved (script does NOT touch):
#   data/phase3_5/       — Mar 6 validated baseline
#   data/phase3_10/      — Mar 6, run after Phase 3.5
#   data/phase3_11/      — Mar 6, run after Phase 3.5
#   data/phase2_*/       — Feb 8 feature discovery (baseline-independent)
#   data/phase0/         — Feb 10 dataset splits
#   data/phase0_1/       — Feb 10 dataset splits
#   data/phase1_0/       — Feb 8 code generation
#   data/phase*_gemma9b* — Gemma-9B (clean baseline)
#   data/phase*_llama*   — LLAMA (independent)
#
# Config: google/gemma-2-2b + mbpp (defaults, no flags needed)
#
# Usage:
#   screen -S gemma2b_resync
#   bash scripts/run_gemma2b_resync.sh

set -e

LOG_FILE="scripts/gemma2b_resync.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

PIPELINE_START=$(date +%s)

echo "============================================================"
echo "Gemma-2-2B Re-sync (CUBLAS-fixed baseline, from Phase 3.5)"
echo "Started: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "Preserved: Phase 3.5 (Mar 6), Phase 3.10/3.11, Phase 2.x, 0.x, 1.x"
echo "============================================================"

# OPTIONAL: Re-run Phase 3.5 for 100% clean CUBLAS-fixed baseline
# (validated at 0 flips/50 problems; uncomment only if re-running everything)
# rm -rf data/phase3_5 data/phase3_5/activations data/phase3_5/parallel_checkpoints
# run_phase "Phase 3.5: Temperature robustness (CUBLAS-fixed baseline)" \
#     phase 3.5 --parallel 4

# Helper: wraps `python3 run.py` with per-phase timing
run_phase() {
    local description="$1"
    shift
    local phase_start=$(date +%s)

    echo ""
    echo "============================================================"
    echo "$description"
    echo "Started: $(date)"
    echo "============================================================"

    python3 run.py "$@"

    local phase_end=$(date +%s)
    local elapsed=$(( phase_end - phase_start ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    echo "Completed in ${mins}m ${secs}s"
}

# ============================================================
# STAGE 3: Statistical Validation (SAE)
# ============================================================
echo ""
echo "############ STAGE 3: Statistical Validation (SAE) ############"

rm -rf data/phase3_6
run_phase "Phase 3.6: Hyperparameter baseline" \
    phase 3.6 --parallel 4

rm -rf data/phase3_8
run_phase "Phase 3.8: AUROC/F1 evaluation" \
    phase 3.8

rm -rf data/phase3_12
run_phase "Phase 3.12: Difficulty-based AUROC" \
    phase 3.12

# ============================================================
# STAGE 4: Steering Pipeline (SAE)
# ============================================================
echo ""
echo "############ STAGE 4: Steering Pipeline (SAE) ############"

rm -rf data/phase4_5
run_phase "Phase 4.5: Coefficient grid search" \
    phase 4.5 --parallel 4

rm -rf data/phase4_6
run_phase "Phase 4.6: Golden section refinement" \
    phase 4.6 --parallel 4

rm -rf data/phase4_7
run_phase "Phase 4.7: Coefficient visualization" \
    phase 4.7

rm -rf data/phase4_8
run_phase "Phase 4.8: Steering effect analysis" \
    phase 4.8 --parallel 4

rm -rf data/phase4_9
run_phase "Phase 4.9: Best latent selection (SAE-only)" \
    phase 4.9

rm -rf data/phase4_10
run_phase "Phase 4.10: Zero-disc feature selection" \
    phase 4.10

rm -rf data/phase4_12
run_phase "Phase 4.12: Zero-disc steering" \
    phase 4.12 --parallel 4

rm -rf data/phase4_14
run_phase "Phase 4.14: Statistical significance" \
    phase 4.14

rm -rf data/phase4_16
run_phase "Phase 4.16: Difficulty-stratified steering" \
    phase 4.16

# ============================================================
# STAGE 5: Weight Orthogonalization (SAE)
# ============================================================
echo ""
echo "############ STAGE 5: Weight Orthogonalization (SAE) ############"

rm -rf data/phase5_3
run_phase "Phase 5.3: Weight orthogonalization" \
    phase 5.3 --parallel 4

rm -rf data/phase5_6
run_phase "Phase 5.6: Zero-disc orthogonalization (SAE-only)" \
    phase 5.6 --parallel 4

rm -rf data/phase5_9
run_phase "Phase 5.9: Orthogonalization significance" \
    phase 5.9

# ============================================================
# STAGE 6: Attention Pattern Analysis (SAE)
# ============================================================
echo ""
echo "############ STAGE 6: Attention Pattern Analysis (SAE) ############"

rm -rf data/phase6_3
run_phase "Phase 6.3: Attention pattern analysis" \
    phase 6.3

# ============================================================
# STAGE 7: Instruction-Tuned Model (SAE)
# ============================================================
echo ""
echo "############ STAGE 7: Instruction-Tuned Model (SAE) ############"

rm -rf data/phase7_3
run_phase "Phase 7.3: Instruct baseline" \
    phase 7.3 --parallel 4

rm -rf data/phase7_6
run_phase "Phase 7.6: Instruct steering" \
    phase 7.6 --parallel 4

rm -rf data/phase7_7
run_phase "Phase 7.7: Instruct zero-disc (SAE-only)" \
    phase 7.7 --parallel 4

rm -rf data/phase7_9
run_phase "Phase 7.9: Universality analysis" \
    phase 7.9

rm -rf data/phase7_12
run_phase "Phase 7.12: Instruct AUROC/F1" \
    phase 7.12

# ============================================================
# STAGE 8: Selective Steering (SAE)
# ============================================================
echo ""
echo "############ STAGE 8: Selective Steering (SAE) ############"

rm -rf data/phase8_1
run_phase "Phase 8.1: Percentile threshold calculator" \
    phase 8.1

rm -rf data/phase8_2
run_phase "Phase 8.2: Percentile threshold optimizer" \
    phase 8.2 --parallel 4

rm -rf data/phase8_3
run_phase "Phase 8.3: Selective steering" \
    phase 8.3 --parallel 4

rm -rf data/phase8_7
run_phase "Phase 8.7: Threshold search visualization" \
    phase 8.7

# ============================================================
# STAGE 9: Probe Prediction + Steering (probe_logreg / probe_mass_mean)
# ============================================================
echo ""
echo "############ STAGE 9: Probe Prediction + Steering ############"

rm -rf data/phase3_8_probe
run_phase "Phase 3.8: Probe AUROC/F1 (logreg)" \
    phase 3.8 --direction-source probe_logreg

rm -rf data/phase4_5_probe
run_phase "Phase 4.5: Probe coefficient search (mass_mean)" \
    phase 4.5 --direction-source probe_mass_mean --parallel 4

rm -rf data/phase4_6_probe
run_phase "Phase 4.6: Probe golden section (mass_mean)" \
    phase 4.6 --direction-source probe_mass_mean --parallel 4

rm -rf data/phase4_8_probe
run_phase "Phase 4.8: Probe steering effect (mass_mean)" \
    phase 4.8 --direction-source probe_mass_mean --parallel 4

# ============================================================
# STAGE 10: Probe Orthogonalization
# ============================================================
echo ""
echo "############ STAGE 10: Probe Orthogonalization ############"

rm -rf data/phase5_3_probe
run_phase "Phase 5.3: Probe weight orthogonalization (mass_mean)" \
    phase 5.3 --direction-source probe_mass_mean --parallel 4

# ============================================================
# STAGE 11: Probe Attention
# ============================================================
echo ""
echo "############ STAGE 11: Probe Attention ############"

rm -rf data/phase6_3_probe
run_phase "Phase 6.3: Probe attention pattern analysis (mass_mean)" \
    phase 6.3 --direction-source probe_mass_mean

# ============================================================
# STAGE 12: Probe Instruction-Tuned
# ============================================================
echo ""
echo "############ STAGE 12: Probe Instruction-Tuned ############"

rm -rf data/phase7_6_probe
run_phase "Phase 7.6: Probe instruct steering (mass_mean)" \
    phase 7.6 --direction-source probe_mass_mean --parallel 4

rm -rf data/phase7_12_probe
run_phase "Phase 7.12: Probe instruct AUROC/F1 (logreg)" \
    phase 7.12 --direction-source probe_logreg

# ============================================================
# STAGE 13: Probe Selective Steering
# ============================================================
echo ""
echo "############ STAGE 13: Probe Selective Steering ############"

rm -rf data/phase8_1_probe
run_phase "Phase 8.1: Probe threshold calculator (logreg)" \
    phase 8.1 --direction-source probe_logreg

rm -rf data/phase8_2_probe
run_phase "Phase 8.2: Probe threshold optimizer (logreg)" \
    phase 8.2 --direction-source probe_logreg --parallel 4

rm -rf data/phase8_3_probe
run_phase "Phase 8.3: Probe selective steering (logreg)" \
    phase 8.3 --direction-source probe_logreg --parallel 4

# ============================================================
# STAGE 14: Final Analysis
# ============================================================
echo ""
echo "############ STAGE 14: Final Analysis ############"

rm -rf data/phase9_5
run_phase "Phase 9.5: Combined weight orthogonalization + activation steering" \
    phase 9.5

rm -rf data/phase11_5
run_phase "Phase 11.5: Error type summary aggregator" \
    phase 11.5

# ============================================================
# DONE
# ============================================================
PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$(( PIPELINE_END - PIPELINE_START ))
TOTAL_HOURS=$(( TOTAL_ELAPSED / 3600 ))
TOTAL_MINS=$(( (TOTAL_ELAPSED % 3600) / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))

echo ""
echo "============================================================"
echo "Re-sync Complete! (CUBLAS-fixed baseline from Phase 3.5)"
echo "Finished: $(date)"
echo "Total elapsed: ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "============================================================"
