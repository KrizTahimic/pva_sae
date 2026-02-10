#!/bin/bash
# Full production pipeline for Gemma-2-2B (SAE + Probe)
#
# Reruns everything downstream of Phase 2.2 after the activation hook fix
# (register_forward_hook → register_forward_pre_hook for resid_pre).
#
# Prerequisites:
#   - Phase 0, 0.1, 1 data already exists (not affected by hook fix)
#   - Phase 2.6 probe data already exists (trained on Phase 1 activations)
#   - Config: google/gemma-2-2b + mbpp (defaults, no flags needed)
#
# Usage:
#   screen -S gemma2b
#   bash scripts/run_gemma2b_full_pipeline.sh
#
# Total: 42 phase runs (31 SAE + 11 probe)

set -e

LOG_FILE="scripts/gemma2b_pipeline.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

PIPELINE_START=$(date +%s)

echo "============================================================"
echo "Gemma-2-2B Full Pipeline (SAE + Probe)"
echo "Started: $(date)"
echo "============================================================"

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
# STAGE 1: Feature Discovery (SAE latent selection)
# ============================================================
echo ""
echo "############ STAGE 1: Feature Discovery ############"

run_phase "Phase 2.2: Pile activation caching" \
    phase 2.2 --parallel 4

run_phase "Phase 2.3: Pile SAE frequency computation" \
    phase 2.3

run_phase "Phase 2.5: SAE analysis + pile filtering" \
    phase 2.5

run_phase "Phase 2.10: T-statistic latent selection" \
    phase 2.10

run_phase "Phase 2.13: Threshold sensitivity analysis" \
    phase 2.13

run_phase "Phase 2.20: Latent landscape visualization" \
    phase 2.20

# ============================================================
# STAGE 2: Statistical Validation (SAE)
# ============================================================
echo ""
echo "############ STAGE 2: Statistical Validation (SAE) ############"

run_phase "Phase 3.8: AUROC/F1 evaluation" \
    phase 3.8

run_phase "Phase 3.10: Temperature-based AUROC" \
    phase 3.10

run_phase "Phase 3.11: Temperature trends visualization" \
    phase 3.11

run_phase "Phase 3.12: Difficulty-based AUROC" \
    phase 3.12

# ============================================================
# STAGE 3: Steering Pipeline (SAE)
# ============================================================
echo ""
echo "############ STAGE 3: Steering Pipeline (SAE) ############"

run_phase "Phase 4.5: Coefficient grid search" \
    phase 4.5 --parallel 4

run_phase "Phase 4.6: Golden section refinement" \
    phase 4.6 --parallel 4

run_phase "Phase 4.7: Coefficient visualization" \
    phase 4.7

run_phase "Phase 4.8: Steering effect analysis" \
    phase 4.8 --parallel 4

run_phase "Phase 4.9: Best latent selection (SAE-only)" \
    phase 4.9

run_phase "Phase 4.10: Zero-disc feature selection" \
    phase 4.10

run_phase "Phase 4.12: Zero-disc steering" \
    phase 4.12 --parallel 4

run_phase "Phase 4.14: Statistical significance" \
    phase 4.14

run_phase "Phase 4.16: Difficulty-stratified steering" \
    phase 4.16

# ============================================================
# STAGE 4: Weight Orthogonalization (SAE)
# ============================================================
echo ""
echo "############ STAGE 4: Weight Orthogonalization (SAE) ############"

run_phase "Phase 5.3: Weight orthogonalization" \
    phase 5.3 --parallel 4

run_phase "Phase 5.6: Zero-disc orthogonalization (SAE-only)" \
    phase 5.6 --parallel 4

run_phase "Phase 5.9: Orthogonalization significance" \
    phase 5.9

# ============================================================
# STAGE 5: Instruction-Tuned Model (SAE)
# ============================================================
echo ""
echo "############ STAGE 5: Instruction-Tuned Model (SAE) ############"

run_phase "Phase 7.3: Instruct baseline" \
    phase 7.3 --parallel 4

run_phase "Phase 7.6: Instruct steering" \
    phase 7.6 --parallel 4

run_phase "Phase 7.7: Instruct zero-disc (SAE-only)" \
    phase 7.7 --parallel 4

run_phase "Phase 7.9: Universality analysis" \
    phase 7.9

run_phase "Phase 7.12: Instruct AUROC/F1" \
    phase 7.12

# ============================================================
# STAGE 6: Selective Steering (SAE)
# ============================================================
echo ""
echo "############ STAGE 6: Selective Steering (SAE) ############"

run_phase "Phase 8.1: Percentile threshold calculator" \
    phase 8.1

run_phase "Phase 8.2: Percentile threshold optimizer" \
    phase 8.2 --parallel 4

run_phase "Phase 8.3: Selective steering" \
    phase 8.3 --parallel 4

run_phase "Phase 8.7: Threshold search visualization" \
    phase 8.7

# ============================================================
# STAGE 7: Probe Prediction (probe_logreg)
# ============================================================
echo ""
echo "############ STAGE 7: Probe Prediction ############"

run_phase "Phase 3.8: Probe AUROC/F1 (logreg)" \
    phase 3.8 --direction-source probe_logreg

run_phase "Phase 3.10: Probe temperature AUROC (logreg)" \
    phase 3.10 --direction-source probe_logreg

# ============================================================
# STAGE 8: Probe Steering (probe_mass_mean)
# ============================================================
echo ""
echo "############ STAGE 8: Probe Steering ############"

run_phase "Phase 4.5: Probe coefficient search (mass_mean)" \
    phase 4.5 --direction-source probe_mass_mean --parallel 4

run_phase "Phase 4.6: Probe golden section (mass_mean)" \
    phase 4.6 --direction-source probe_mass_mean --parallel 4

run_phase "Phase 4.8: Probe steering effect (mass_mean)" \
    phase 4.8 --direction-source probe_mass_mean --parallel 4

# ============================================================
# STAGE 9: Probe Orthogonalization
# ============================================================
echo ""
echo "############ STAGE 9: Probe Orthogonalization ############"

run_phase "Phase 5.3: Probe weight orthogonalization (mass_mean)" \
    phase 5.3 --direction-source probe_mass_mean --parallel 4

# ============================================================
# STAGE 10: Probe Instruction-Tuned
# ============================================================
echo ""
echo "############ STAGE 10: Probe Instruction-Tuned ############"

run_phase "Phase 7.6: Probe instruct steering (mass_mean)" \
    phase 7.6 --direction-source probe_mass_mean --parallel 4

run_phase "Phase 7.12: Probe instruct AUROC/F1 (logreg)" \
    phase 7.12 --direction-source probe_logreg

# ============================================================
# STAGE 11: Probe Selective Steering
# ============================================================
echo ""
echo "############ STAGE 11: Probe Selective Steering ############"

run_phase "Phase 8.1: Probe threshold calculator (logreg)" \
    phase 8.1 --direction-source probe_logreg

run_phase "Phase 8.2: Probe threshold optimizer (logreg)" \
    phase 8.2 --direction-source probe_logreg --parallel 4

run_phase "Phase 8.3: Probe selective steering (logreg)" \
    phase 8.3 --direction-source probe_logreg --parallel 4

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
echo "Pipeline Complete!"
echo "Finished: $(date)"
echo "Total elapsed: ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "============================================================"
