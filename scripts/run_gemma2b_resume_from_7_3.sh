#!/bin/bash
# Resume Gemma-2-2B pipeline from Phase 7.3 onward (SAE + Probe).
# Phases 0 through 5.9, 6.3 already completed.
#
# Config: google/gemma-2-2b + mbpp (defaults, no flags needed)
#
# Usage:
#   screen -S gemma2b
#   bash scripts/run_gemma2b_resume_from_7_3.sh

set -e

LOG_FILE="scripts/gemma2b_resume_from_7_3.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

PIPELINE_START=$(date +%s)

echo "============================================================"
echo "Gemma-2-2B Pipeline — Resume from Phase 7.3"
echo "Started: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "Skipped: Phases 0-5.9, 6.3 (already completed)"
echo "============================================================"

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
# STAGE 6: Instruction-Tuned Model (SAE)
# ============================================================
echo ""
echo "############ STAGE 6: Instruction-Tuned Model (SAE) ############"

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
# STAGE 7: Selective Steering (SAE)
# ============================================================
echo ""
echo "############ STAGE 7: Selective Steering (SAE) ############"

run_phase "Phase 8.1: Percentile threshold calculator" \
    phase 8.1

run_phase "Phase 8.2: Percentile threshold optimizer" \
    phase 8.2 --parallel 4

run_phase "Phase 8.3: Selective steering" \
    phase 8.3 --parallel 4

run_phase "Phase 8.7: Threshold search visualization" \
    phase 8.7

# ============================================================
# STAGE 8: Probe Prediction (probe_logreg)
# ============================================================
echo ""
echo "############ STAGE 8: Probe Prediction ############"

run_phase "Phase 3.8: Probe AUROC/F1 (logreg)" \
    phase 3.8 --direction-source probe_logreg

# ============================================================
# STAGE 9: Probe Steering (probe_mass_mean)
# ============================================================
echo ""
echo "############ STAGE 9: Probe Steering ############"

run_phase "Phase 4.5: Probe coefficient search (mass_mean)" \
    phase 4.5 --direction-source probe_mass_mean --parallel 4

run_phase "Phase 4.6: Probe golden section (mass_mean)" \
    phase 4.6 --direction-source probe_mass_mean --parallel 4

run_phase "Phase 4.8: Probe steering effect (mass_mean)" \
    phase 4.8 --direction-source probe_mass_mean --parallel 4

run_phase "Phase 6.3: Probe attention pattern analysis (mass_mean)" \
    phase 6.3 --direction-source probe_mass_mean

# ============================================================
# STAGE 10: Probe Orthogonalization
# ============================================================
echo ""
echo "############ STAGE 10: Probe Orthogonalization ############"

run_phase "Phase 5.3: Probe weight orthogonalization (mass_mean)" \
    phase 5.3 --direction-source probe_mass_mean --parallel 4

# ============================================================
# STAGE 11: Probe Instruction-Tuned
# ============================================================
echo ""
echo "############ STAGE 11: Probe Instruction-Tuned ############"

run_phase "Phase 7.6: Probe instruct steering (mass_mean)" \
    phase 7.6 --direction-source probe_mass_mean --parallel 4

run_phase "Phase 7.12: Probe instruct AUROC/F1 (logreg)" \
    phase 7.12 --direction-source probe_logreg

# ============================================================
# STAGE 12: Probe Selective Steering
# ============================================================
echo ""
echo "############ STAGE 12: Probe Selective Steering ############"

run_phase "Phase 8.1: Probe threshold calculator (logreg)" \
    phase 8.1 --direction-source probe_logreg

run_phase "Phase 8.2: Probe threshold optimizer (logreg)" \
    phase 8.2 --direction-source probe_logreg --parallel 4

run_phase "Phase 8.3: Probe selective steering (logreg)" \
    phase 8.3 --direction-source probe_logreg --parallel 4

# ============================================================
# STAGE 13: Error Type Summary
# ============================================================
echo ""
echo "############ STAGE 13: Error Type Summary ############"

run_phase "Phase 9.5: Error type summary" \
    phase 9.5

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
echo "Pipeline Complete! (resumed from Phase 7.3)"
echo "Finished: $(date)"
echo "Total elapsed: ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "============================================================"
