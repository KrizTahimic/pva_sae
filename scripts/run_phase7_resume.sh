#!/bin/bash
# Resume from Phase 7.6 Gemma probe (Phase 7.3 + 7.12 Gemma already done).
# Fixes: also patches phase7_6_model_name to Gemma instruct (was missing in original script).

set -e

LOG_FILE="scripts/phase7_rerun.log"
exec > >(tee -a "$LOG_FILE") 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

PIPELINE_START=$(date +%s)

echo "============================================================"
echo "Phase 7 Resume (from 7.6 Gemma probe)"
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

restore_config() {
    echo "Restoring config.py..."
    git checkout common/config.py
}

# ============================================================
# GEMMA — Phase 7.6 probe only (7.3 + 7.12 already done)
# ============================================================
echo ""
echo "############ GEMMA — Phase 7.6 probe ############"

echo "Patching config.py for Gemma instruct..."
sed -i 's|phase7_6_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"|phase7_6_model_name: str = "google/gemma-2-2b-it"|' common/config.py
trap restore_config EXIT

run_phase "Phase 7.6 probe: Gemma instruct steering (mass_mean, Phase 4.5/4.6 layers)" \
    phase 7.6 --direction-source probe_mass_mean --parallel 4

restore_config
trap - EXIT

# ============================================================
# LLAMA (meta-llama/Llama-3.1-8B + meta-llama/Llama-3.1-8B-Instruct)
# ============================================================
echo ""
echo "############ LLAMA ############"

run_phase "Phase 7.3: LLAMA instruct baseline (captures Phase 4.9 + Phase 3.8 layers)" \
    phase 7.3 --model meta-llama/Llama-3.1-8B --parallel 4

run_phase "Phase 7.12: LLAMA instruct AUROC/F1 (SAE, Phase 3.8 latent)" \
    phase 7.12 --model meta-llama/Llama-3.1-8B

run_phase "Phase 7.12: LLAMA instruct AUROC/F1 (probe, Phase 3.8 layer)" \
    phase 7.12 --model meta-llama/Llama-3.1-8B --direction-source probe_logreg

run_phase "Phase 7.6 probe: LLAMA instruct steering (mass_mean, Phase 4.5/4.6 layers)" \
    phase 7.6 --model meta-llama/Llama-3.1-8B --direction-source probe_mass_mean --parallel 4

# ============================================================
# PHASE 8.2 + 8.3 PROBE (Phase 4.5/4.6 layers)
# ============================================================
echo ""
echo "############ PHASE 8 PROBE RE-RUN ############"

run_phase "Phase 8.2: Gemma threshold search (probe, Phase 4.5/4.6 layers)" \
    phase 8.2 --direction-source probe_mass_mean --parallel 4

run_phase "Phase 8.3: Gemma selective steering (probe, Phase 4.5/4.6 layers)" \
    phase 8.3 --direction-source probe_mass_mean --parallel 4

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
echo "Resume complete!"
echo "Finished: $(date)"
echo "Total elapsed: $(( TOTAL_ELAPSED / 3600 ))h $(( (TOTAL_ELAPSED % 3600) / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "============================================================"
