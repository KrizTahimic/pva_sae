#!/bin/bash
# LLAMA-3.1-8B resume from Stage 14 (Probe prediction onward).
# Prereqs: Stages 6-13 complete (through phase6_3_llama_probe).
#
# Runs in order:
#   Stage 14: Phase 3.5 temp fill  (generate missing temps 0.4-1.4)
#             Phase 3.10 probe prediction
#   Stage 15: Probe instruct    (7.6 probe → 7.12 probe)
#   Stage 16: Probe selective   (8.1 probe → 8.2 probe → 8.3 probe)
#   Stage 17: Summary           (9.5)
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
INSTRUCT_MODEL="meta-llama/Llama-3.1-8B-Instruct"

PIPELINE_START=$(date +%s)

echo "============================================================"
echo "LLAMA-3.1-8B Resume from Stage 14"
echo "Started: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "Model: $MODEL"
echo "============================================================"

# Patch config.py for LLAMA instruct model (phases 7.x read from config fields)
echo "Patching config.py instruct model settings for LLAMA..."
sed -i 's|phase7_3_model_name: str = "google/gemma-2-2b-it"|phase7_3_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"|' common/config.py
sed -i 's|phase7_6_model_name: str = "google/gemma-2-2b-it"|phase7_6_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"|' common/config.py
sed -i 's|phase7_7_model_name: str = "google/gemma-2-2b-it"|phase7_7_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"|' common/config.py

# Patch temperature_variation_temps to generate missing temps (0.4-1.4)
echo "Patching config.py to generate missing temperature parquets..."
sed -i 's|temperature_variation_temps: list\[float\] = field(default_factory=lambda: \[0\.0\])|temperature_variation_temps: list[float] = field(default_factory=lambda: [0.4, 0.6, 0.8, 1.0, 1.2, 1.4])|' common/config.py

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
# STAGE 14: Probe Prediction
# (Phase 3.5 temperature fill + Phase 3.10)
# ============================================================
echo ""
echo "############ STAGE 14: Probe Prediction ############"

run_phase "Phase 3.5: Temperature generation (missing temps 0.4-1.4)" \
    phase 3.5 --model "$MODEL" --parallel 4

run_phase "Phase 3.10: Temperature AUROC (probe_logreg)" \
    phase 3.10 --model "$MODEL" --direction-source probe_logreg

# ============================================================
# STAGE 15: Probe Instruction-Tuned Model
# ============================================================
echo ""
echo "############ STAGE 15: Probe Instruction-Tuned Model ############"

run_phase "Phase 7.6: Instruct steering (probe_mass_mean)" \
    phase 7.6 --model "$MODEL" --direction-source probe_mass_mean --parallel 4

run_phase "Phase 7.12: Instruct AUROC/F1 (probe_logreg)" \
    phase 7.12 --model "$MODEL" --direction-source probe_logreg

# ============================================================
# STAGE 16: Probe Selective Steering
# ============================================================
echo ""
echo "############ STAGE 16: Probe Selective Steering ############"

run_phase "Phase 8.1: Percentile threshold calculator (probe_logreg)" \
    phase 8.1 --model "$MODEL" --direction-source probe_logreg

run_phase "Phase 8.2: Percentile threshold optimizer (probe_logreg)" \
    phase 8.2 --model "$MODEL" --direction-source probe_logreg --parallel 4

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
echo "LLAMA Stage 14+ Pipeline Complete!"
echo "Finished: $(date)"
echo "Total elapsed: ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "============================================================"
