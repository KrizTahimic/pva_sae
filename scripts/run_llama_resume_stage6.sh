#!/bin/bash
# LLAMA-3.1-8B resume from Stage 6 (SAE steering onward).
# Prereqs: Phases 1–3.8 already complete for LLAMA (phase1_0_llama through phase3_8_llama).
# Phase 4.5 probe also already complete (phase4_5_llama_probe).
#
# Runs in order:
#   Stage 6:  SAE steering      (4.5 → 4.9 → 4.10 → 4.12 → 4.14 → 4.16 + viz)
#   Stage 7:  SAE ortho         (5.3 → 5.6 → 5.9)
#   Stage 8:  SAE attention     (6.3)
#   Stage 9:  SAE instruct      (7.3 → 7.6 → 7.7 → 7.9 → 7.12)
#   Stage 10: SAE selective     (8.1 → 8.2 → 8.3 → 8.7)
#   Stage 11: Probe steering    (4.6 → 4.8) [4.5 probe already done]
#   Stage 12: Probe ortho       (5.3 probe)
#   Stage 13: Probe attention   (6.3 probe)
#   Stage 14: Probe prediction  (3.10 probe_logreg)
#   Stage 15: Probe instruct    (7.6 probe → 7.12 probe)
#   Stage 16: Probe selective   (8.1 probe → 8.2 probe → 8.3 probe)
#   Stage 17: Summary           (9.5)
#
# Usage:
#   screen -dmS llama_stage6 bash scripts/run_llama_resume_stage6.sh

set -e

LOG_FILE="scripts/run_llama_resume_stage6.log"
exec > >(tee -a "$LOG_FILE") 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

cd /home/kriz.tahimic/sae-code-correctness

MODEL="meta-llama/Llama-3.1-8B"
INSTRUCT_MODEL="meta-llama/Llama-3.1-8B-Instruct"

PIPELINE_START=$(date +%s)

echo "============================================================"
echo "LLAMA-3.1-8B Resume from Stage 6"
echo "Started: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "Model: $MODEL"
echo "============================================================"

# Patch config.py for LLAMA instruct model (phases 7.x read from config fields)
echo "Patching config.py instruct model settings for LLAMA..."
sed -i 's|phase7_3_model_name: str = "google/gemma-2-2b-it"|phase7_3_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"|' common/config.py
sed -i 's|phase7_6_model_name: str = "google/gemma-2-2b-it"|phase7_6_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"|' common/config.py
sed -i 's|phase7_7_model_name: str = "google/gemma-2-2b-it"|phase7_7_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"|' common/config.py

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
# STAGE 6: SAE Steering Pipeline
# ============================================================
echo ""
echo "############ STAGE 6: SAE Steering Pipeline ############"

run_phase "Phase 4.5: Coefficient grid search (SAE)" \
    phase 4.5 --model "$MODEL" --parallel 4

run_phase "Phase 4.6: Golden section refinement (SAE)" \
    phase 4.6 --model "$MODEL" --parallel 4

run_phase "Phase 4.7: Coefficient visualization (SAE)" \
    phase 4.7 --model "$MODEL"

run_phase "Phase 4.8: Steering effect analysis (SAE)" \
    phase 4.8 --model "$MODEL" --parallel 4

run_phase "Phase 4.9: Best latent selection (SAE-only)" \
    phase 4.9 --model "$MODEL"

run_phase "Phase 4.10: Zero-disc feature selection" \
    phase 4.10 --model "$MODEL"

run_phase "Phase 4.12: Zero-disc steering" \
    phase 4.12 --model "$MODEL" --parallel 4

run_phase "Phase 4.14: Statistical significance" \
    phase 4.14 --model "$MODEL"

run_phase "Phase 4.16: Difficulty-stratified steering" \
    phase 4.16 --model "$MODEL"

# ============================================================
# STAGE 7: SAE Weight Orthogonalization
# ============================================================
echo ""
echo "############ STAGE 7: SAE Weight Orthogonalization ############"

run_phase "Phase 5.3: Weight orthogonalization (SAE)" \
    phase 5.3 --model "$MODEL" --parallel 4

run_phase "Phase 5.6: Zero-disc orthogonalization (SAE-only)" \
    phase 5.6 --model "$MODEL" --parallel 4

run_phase "Phase 5.9: Orthogonalization significance" \
    phase 5.9 --model "$MODEL"

# ============================================================
# STAGE 8: SAE Attention Analysis
# ============================================================
echo ""
echo "############ STAGE 8: SAE Attention Analysis ############"

run_phase "Phase 6.3: Attention pattern analysis (SAE)" \
    phase 6.3 --model "$MODEL"

# ============================================================
# STAGE 9: SAE Instruction-Tuned Model
# ============================================================
echo ""
echo "############ STAGE 9: SAE Instruction-Tuned Model ############"

run_phase "Phase 7.3: Instruct baseline" \
    phase 7.3 --model "$MODEL" --parallel 4

run_phase "Phase 7.6: Instruct steering (SAE)" \
    phase 7.6 --model "$MODEL" --parallel 4

run_phase "Phase 7.7: Instruct zero-disc (SAE-only)" \
    phase 7.7 --model "$MODEL" --parallel 4

run_phase "Phase 7.9: Universality analysis" \
    phase 7.9 --model "$MODEL"

run_phase "Phase 7.12: Instruct AUROC/F1 (SAE)" \
    phase 7.12 --model "$MODEL"

# ============================================================
# STAGE 10: SAE Selective Steering
# ============================================================
echo ""
echo "############ STAGE 10: SAE Selective Steering ############"

run_phase "Phase 8.1: Percentile threshold calculator (SAE)" \
    phase 8.1 --model "$MODEL"

run_phase "Phase 8.2: Percentile threshold optimizer (SAE)" \
    phase 8.2 --model "$MODEL" --parallel 4

run_phase "Phase 8.3: Selective steering (SAE)" \
    phase 8.3 --model "$MODEL" --parallel 4

run_phase "Phase 8.7: Threshold search visualization (SAE)" \
    phase 8.7 --model "$MODEL"

# ============================================================
# STAGE 11: Probe Steering (4.5 already done)
# ============================================================
echo ""
echo "############ STAGE 11: Probe Steering ############"

run_phase "Phase 4.6: Golden section refinement (probe_mass_mean)" \
    phase 4.6 --model "$MODEL" --direction-source probe_mass_mean --parallel 4

run_phase "Phase 4.8: Steering effect analysis (probe_mass_mean)" \
    phase 4.8 --model "$MODEL" --direction-source probe_mass_mean --parallel 4

# ============================================================
# STAGE 12: Probe Weight Orthogonalization
# ============================================================
echo ""
echo "############ STAGE 12: Probe Weight Orthogonalization ############"

run_phase "Phase 5.3: Weight orthogonalization (probe_mass_mean)" \
    phase 5.3 --model "$MODEL" --direction-source probe_mass_mean --parallel 4

# ============================================================
# STAGE 13: Probe Attention Analysis
# ============================================================
echo ""
echo "############ STAGE 13: Probe Attention Analysis ############"

run_phase "Phase 6.3: Attention pattern analysis (probe_mass_mean)" \
    phase 6.3 --model "$MODEL" --direction-source probe_mass_mean

# ============================================================
# STAGE 14: Probe Prediction
# ============================================================
echo ""
echo "############ STAGE 14: Probe Prediction ############"

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
echo "LLAMA Stage 6+ Pipeline Complete!"
echo "Finished: $(date)"
echo "Total elapsed: ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "============================================================"
