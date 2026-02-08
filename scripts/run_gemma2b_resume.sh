#!/bin/bash
# Resume Gemma-2-2B pipeline after pickle bug fix in retry_utils.py
#
# Reruns phases degraded by the nested-Process pickle error:
#   - 4.12: silently excluded all tasks (zero results)
#   - 5.3, 5.6: all tasks marked as failures (placeholder results)
#   - 5.9: depends on 5.3/5.6 data
# Then continues with all phases that weren't reached (7.3 onward).
#
# Prerequisites:
#   - Phases 0–4.10 completed successfully (not affected by bug)
#   - Config: google/gemma-2-2b + mbpp (defaults)
#
# Usage:
#   screen -S gemma2b
#   bash scripts/run_gemma2b_resume.sh

set -e

LOG_FILE="scripts/gemma2b_resume.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

PIPELINE_START=$(date +%s)

echo "============================================================"
echo "Gemma-2-2B Resume Pipeline (post pickle-bug fix)"
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
# RERUN: Degraded SAE steering phases + dependents
# ============================================================
echo ""
echo "############ RERUN: Degraded Phases (pickle bug) ############"

run_phase "Phase 4.12: Zero-disc steering (RERUN)" \
    phase 4.12 --parallel 4

run_phase "Phase 4.14: Statistical significance (RERUN - depends on 4.12)" \
    phase 4.14

run_phase "Phase 4.16: Difficulty-stratified steering (RERUN)" \
    phase 4.16

# ============================================================
# RERUN: Degraded SAE orthogonalization phases
# ============================================================
echo ""
echo "############ RERUN: Degraded Orthogonalization ############"

run_phase "Phase 5.3: Weight orthogonalization (RERUN)" \
    phase 5.3 --parallel 4

run_phase "Phase 5.6: Zero-disc orthogonalization (RERUN)" \
    phase 5.6 --parallel 4

run_phase "Phase 5.9: Orthogonalization significance (RERUN)" \
    phase 5.9

# ============================================================
# STAGE 5: Instruction-Tuned Model (SAE) - not yet started
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
# STAGE 6: Selective Steering (SAE) - not yet started
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
# STAGE 7: Probe Prediction (probe_logreg) - not yet started
# ============================================================
echo ""
echo "############ STAGE 7: Probe Prediction ############"

run_phase "Phase 3.8: Probe AUROC/F1 (logreg)" \
    phase 3.8 --direction-source probe_logreg

run_phase "Phase 3.10: Probe temperature AUROC (logreg)" \
    phase 3.10 --direction-source probe_logreg

# ============================================================
# STAGE 8: Probe Steering (probe_mass_mean) - not yet started
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
# STAGE 9: Probe Orthogonalization - not yet started
# ============================================================
echo ""
echo "############ STAGE 9: Probe Orthogonalization ############"

run_phase "Phase 5.3: Probe weight orthogonalization (mass_mean)" \
    phase 5.3 --direction-source probe_mass_mean --parallel 4

# ============================================================
# STAGE 10: Probe Instruction-Tuned - not yet started
# ============================================================
echo ""
echo "############ STAGE 10: Probe Instruction-Tuned ############"

run_phase "Phase 7.6: Probe instruct steering (mass_mean)" \
    phase 7.6 --direction-source probe_mass_mean --parallel 4

run_phase "Phase 7.12: Probe instruct AUROC/F1 (logreg)" \
    phase 7.12 --direction-source probe_logreg

# ============================================================
# STAGE 11: Probe Selective Steering - not yet started
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
echo "Resume Pipeline Complete!"
echo "Finished: $(date)"
echo "Total elapsed: ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "============================================================"
