#!/bin/bash
# Re-run Phase 7.3, 7.12, and 7.6 probe for Gemma and LLAMA after fixing layer selection.
#
# Fixes applied:
#   Phase 7.3:  now captures Phase 3.8 AUROC-best layers (not just Phase 4.9).
#   Phase 7.12: now uses Phase 3.8 latent/layer (not Phase 4.9/2.6).
#   Phase 7.6 probe: now uses Phase 4.5/4.6 steering-validated layers (not Phase 2.6 cross-val L23).
#
# Prerequisite: data/phase7_3/, data/phase7_3_llama/, data/phase7_12/,
#               data/phase7_12_llama/, data/phase7_12_llama_probe/,
#               data/phase7_6_llama_probe/ deleted.
#
# Usage:
#   screen -S phase7_rerun
#   bash scripts/run_phase7_rerun.sh

set -e

LOG_FILE="scripts/phase7_rerun.log"
exec > >(tee -a "$LOG_FILE") 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

PIPELINE_START=$(date +%s)

echo "============================================================"
echo "Phase 7.3 + 7.12 Re-run (Gemma + LLAMA)"
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
# GEMMA (google/gemma-2-2b + google/gemma-2-2b-it)
# ============================================================
echo ""
echo "############ GEMMA ############"

# Patch phase7_3_model_name to Gemma instruct (committed default is llama)
echo "Patching config.py for Gemma instruct..."
sed -i 's|phase7_3_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"|phase7_3_model_name: str = "google/gemma-2-2b-it"|' common/config.py

restore_config() {
    echo "Restoring config.py..."
    git checkout common/config.py
}
trap restore_config EXIT

run_phase "Phase 7.3: Gemma instruct baseline (captures Phase 4.9 + Phase 3.8 layers)" \
    phase 7.3 --parallel 4

# Restore before 7.12 (7.12 reads only from saved activations, no model load)
restore_config
trap - EXIT

run_phase "Phase 7.12: Gemma instruct AUROC/F1 (SAE, Phase 3.8 latent)" \
    phase 7.12

run_phase "Phase 7.12: Gemma instruct AUROC/F1 (probe, Phase 3.8 layer)" \
    phase 7.12 --direction-source probe_logreg

run_phase "Phase 7.6 probe: Gemma instruct steering (mass_mean, Phase 4.5/4.6 layers)" \
    phase 7.6 --direction-source probe_mass_mean --parallel 4

# ============================================================
# LLAMA (meta-llama/Llama-3.1-8B + meta-llama/Llama-3.1-8B-Instruct)
# ============================================================
echo ""
echo "############ LLAMA ############"

# phase7_3_model_name = "meta-llama/Llama-3.1-8B-Instruct" is already the committed default

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
echo "Re-run complete!"
echo "Finished: $(date)"
echo "Total elapsed: $(( TOTAL_ELAPSED / 3600 ))h $(( (TOTAL_ELAPSED % 3600) / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "============================================================"
