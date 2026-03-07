#!/bin/bash
# Run Phase 8.7 (all 4 variants) and Phase 9.5 (Gemma + LLAMA).
# All phases are visualization/aggregation only — no GPU needed.

set -e

LOG_FILE="scripts/phase89_viz.log"
exec > >(tee -a "$LOG_FILE") 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

PIPELINE_START=$(date +%s)

echo "============================================================"
echo "Phase 8.7 + 9.5 Visualization Run"
echo "Started: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
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
    echo "Completed in $(( elapsed / 60 ))m $(( elapsed % 60 ))s"
}

# ============================================================
# Phase 8.7 — SAE (Gemma + LLAMA)
# ============================================================
echo ""
echo "############ Phase 8.7 — SAE ############"

run_phase "Phase 8.7: Gemma SAE threshold visualization" \
    phase 8.7

run_phase "Phase 8.7: LLAMA SAE threshold visualization" \
    phase 8.7 --model meta-llama/Llama-3.1-8B

# ============================================================
# Phase 8.7 — probe (Gemma + LLAMA)
# ============================================================
echo ""
echo "############ Phase 8.7 — probe ############"

run_phase "Phase 8.7: Gemma probe threshold visualization" \
    phase 8.7 --direction-source probe_mass_mean

run_phase "Phase 8.7: LLAMA probe threshold visualization" \
    phase 8.7 --model meta-llama/Llama-3.1-8B --direction-source probe_mass_mean

# ============================================================
# Phase 9.5 — Gemma + LLAMA
# ============================================================
echo ""
echo "############ Phase 9.5 — error type summary ############"

run_phase "Phase 9.5: Gemma error type summary" \
    phase 9.5

run_phase "Phase 9.5: LLAMA error type summary" \
    phase 9.5 --model meta-llama/Llama-3.1-8B

# ============================================================
# DONE
# ============================================================
PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$(( PIPELINE_END - PIPELINE_START ))

echo ""
echo "============================================================"
echo "Phase 8.7 + 9.5 visualization complete!"
echo "Finished: $(date)"
echo "Total elapsed: $(( TOTAL_ELAPSED / 3600 ))h $(( (TOTAL_ELAPSED % 3600) / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "============================================================"
