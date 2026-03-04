#!/bin/bash
# LLAMA-3.1-8B resume from Stage 16b (Phase 8.3 + 9.5).
# Prereqs: Stages 14-15 + 8.1/8.2 complete.
#
# Runs in order:
#   Stage 16b: Phase 8.3 selective steering (probe_logreg)
#   Stage 17:  Phase 9.5 summary
#
# Usage:
#   screen -dmS llama_stage7 bash scripts/run_llama_resume_stage7.sh

set -e

LOG_FILE="scripts/run_llama_resume_stage7.log"
exec > >(tee -a "$LOG_FILE") 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

cd /home/kriz.tahimic/sae-code-correctness

MODEL="meta-llama/Llama-3.1-8B"

PIPELINE_START=$(date +%s)

echo "============================================================"
echo "LLAMA-3.1-8B Resume from Stage 16b"
echo "Started: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "Model: $MODEL"
echo "============================================================"

# Restore config.py on exit (success or failure)
restore_config() {
    echo "Restoring config.py to Gemma defaults..."
    git checkout common/config.py
}
trap restore_config EXIT

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
# STAGE 16b: Selective Steering (Phase 8.3 only)
# ============================================================
echo ""
echo "############ STAGE 16b: Probe Selective Steering ############"

run_phase "Phase 8.3: Selective steering (probe_logreg)" \
    phase 8.3 --model "$MODEL" --direction-source probe_logreg --parallel 4

# ============================================================
# STAGE 17: Summary
# ============================================================
echo ""
echo "############ STAGE 17: Summary ############"

run_phase "Phase 9.5: Error type summary" \
    phase 9.5 --model "$MODEL"

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
echo "LLAMA Stage 16b+ Pipeline Complete!"
echo "Finished: $(date)"
echo "Total elapsed: ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "============================================================"
