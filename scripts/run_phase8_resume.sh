#!/bin/bash
# Resume from Phase 8.3 Gemma probe.
# Phase 8.1 Gemma probe was missing — run it first, then 8.3 Gemma, then 8.2+8.3 LLAMA.

set -e

LOG_FILE="scripts/phase7_rerun.log"
exec > >(tee -a "$LOG_FILE") 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

PIPELINE_START=$(date +%s)

echo "============================================================"
echo "Phase 8 Probe Resume"
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
# GEMMA — Phase 8.1 probe (missing prerequisite) + 8.3
# ============================================================
echo ""
echo "############ GEMMA — Phase 8.1 + 8.3 probe ############"

run_phase "Phase 8.1: Gemma percentile thresholds (probe)" \
    phase 8.1 --direction-source probe_logreg

run_phase "Phase 8.3: Gemma selective steering (probe, Phase 4.5/4.6 layers)" \
    phase 8.3 --direction-source probe_mass_mean --parallel 4

# ============================================================
# LLAMA — Phase 8.2 + 8.3 probe
# ============================================================
echo ""
echo "############ LLAMA — Phase 8.2 + 8.3 probe ############"

run_phase "Phase 8.2: LLAMA threshold search (probe, Phase 4.5/4.6 layers)" \
    phase 8.2 --model meta-llama/Llama-3.1-8B --direction-source probe_mass_mean --parallel 4

run_phase "Phase 8.3: LLAMA selective steering (probe, Phase 4.5/4.6 layers)" \
    phase 8.3 --model meta-llama/Llama-3.1-8B --direction-source probe_mass_mean --parallel 4

# ============================================================
# DONE
# ============================================================
PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$(( PIPELINE_END - PIPELINE_START ))

echo ""
echo "============================================================"
echo "Phase 8 probe resume complete!"
echo "Finished: $(date)"
echo "Total elapsed: $(( TOTAL_ELAPSED / 3600 ))h $(( (TOTAL_ELAPSED % 3600) / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "============================================================"
