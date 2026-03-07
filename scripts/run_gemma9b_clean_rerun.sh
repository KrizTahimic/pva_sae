#!/bin/bash
# Full Gemma-2-9B pipeline (SAE + Probe) — all phases from scratch.
#
# === PREREQUISITE: Cache swap (run manually before starting pipeline) ===
# Delete Gemma 2B + LLaMA model weights to free space:
#   rm -rf ~/.cache/huggingface/hub/models--google--gemma-2-2b
#   rm -rf ~/.cache/huggingface/hub/models--google--gemma-2-2b-it
#   rm -rf ~/.cache/huggingface/hub/models--google--gemma-scope-2b-pt-res
#   rm -rf ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B
#   rm -rf ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct
#   rm -rf ~/.cache/huggingface/hub/models--meta-llama--llama-scope-lm-8bx32
# Download Gemma 9B (happens automatically on first use, or pre-download):
#   python3 -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('google/gemma-2-9b')"
#   python3 -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('google/gemma-2-9b-it')"
# GemmaScope SAE is downloaded automatically by common/sae_loader.py on first use
#
# Config: google/gemma-2-9b + mbpp
# Phases 0 and 0.1 are SKIPPED — shared data already exists
#
# Usage:
#   screen -S gemma9b
#   bash scripts/run_gemma9b_clean_rerun.sh
#
# Estimated time: ~30-40 hours (Phase 1 ~10h, Phase 4.5/4.6 ~10h, rest ~12h)
# Total: 51 phase runs (33 SAE + 12 probe + phase 0/0.1 skipped — shared data)

set -e

LOG_FILE="scripts/gemma9b_clean_rerun.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

# Patch config for Gemma 9B instruct
sed -i 's|phase7_3_model_name:.*|phase7_3_model_name: str = "google/gemma-2-9b-it"|' common/config.py
sed -i 's|phase7_6_model_name:.*|phase7_6_model_name: str = "google/gemma-2-9b-it"|' common/config.py
sed -i 's|phase7_7_model_name:.*|phase7_7_model_name: str = "google/gemma-2-9b-it"|' common/config.py

# Restore config.py on exit (success or failure)
trap 'git checkout common/config.py' EXIT

PIPELINE_START=$(date +%s)

echo "============================================================"
echo "Gemma-2-9B Clean Full Rerun"
echo "Started: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
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
# STAGE 0: Data Preparation
# ============================================================
# Phases 0 and 0.1 are SKIPPED — shared data already exists

# ============================================================
# STAGE 1: Code Generation + Activations
# ============================================================
echo ""
echo "############ STAGE 1: Code Generation (~10 hours) ############"

run_phase "Phase 1: Code generation + activation extraction" \
    phase 1 --parallel 4 --model google/gemma-2-9b

# ============================================================
# STAGE 2: Feature Discovery (SAE + Probe)
# ============================================================
echo ""
echo "############ STAGE 2: Feature Discovery ############"

run_phase "Phase 2.2: Pile activation caching" \
    phase 2.2 --parallel 4 --model google/gemma-2-9b

run_phase "Phase 2.3: Pile SAE frequency computation" \
    phase 2.3 --model google/gemma-2-9b

run_phase "Phase 2.5: SAE analysis + pile filtering" \
    phase 2.5 --model google/gemma-2-9b

run_phase "Phase 2.6: Probe training (logreg + mass_mean)" \
    phase 2.6 --model google/gemma-2-9b

run_phase "Phase 2.10: T-statistic latent selection" \
    phase 2.10 --model google/gemma-2-9b

run_phase "Phase 2.11: Direction similarity analysis" \
    phase 2.11 --model google/gemma-2-9b

run_phase "Phase 2.13: Threshold sensitivity analysis" \
    phase 2.13 --model google/gemma-2-9b

run_phase "Phase 2.15: Layer-wise analysis visualization" \
    phase 2.15 --model google/gemma-2-9b

run_phase "Phase 2.20: Latent landscape visualization" \
    phase 2.20 --model google/gemma-2-9b

# ============================================================
# STAGE 3: Statistical Validation (SAE)
# ============================================================
echo ""
echo "############ STAGE 3: Statistical Validation (SAE) ############"

run_phase "Phase 3.5: Temperature robustness" \
    phase 3.5 --parallel 4 --model google/gemma-2-9b

run_phase "Phase 3.6: Hyperparameter baseline" \
    phase 3.6 --parallel 4 --model google/gemma-2-9b

run_phase "Phase 3.8: AUROC/F1 evaluation" \
    phase 3.8 --model google/gemma-2-9b

run_phase "Phase 3.10: Temperature-based AUROC" \
    phase 3.10 --model google/gemma-2-9b

run_phase "Phase 3.11: Temperature trends visualization" \
    phase 3.11 --model google/gemma-2-9b

run_phase "Phase 3.12: Difficulty-based AUROC" \
    phase 3.12 --model google/gemma-2-9b

# ============================================================
# STAGE 4: Steering Pipeline (SAE)
# ============================================================
echo ""
echo "############ STAGE 4: Steering Pipeline (SAE) ############"

run_phase "Phase 4.5: Coefficient grid search" \
    phase 4.5 --parallel 4 --model google/gemma-2-9b

run_phase "Phase 4.6: Golden section refinement" \
    phase 4.6 --parallel 4 --model google/gemma-2-9b

run_phase "Phase 4.7: Coefficient visualization" \
    phase 4.7 --model google/gemma-2-9b

run_phase "Phase 4.8: Steering effect analysis" \
    phase 4.8 --parallel 4 --model google/gemma-2-9b

run_phase "Phase 4.9: Best latent selection (SAE-only)" \
    phase 4.9 --model google/gemma-2-9b

run_phase "Phase 4.10: Zero-disc feature selection" \
    phase 4.10 --model google/gemma-2-9b

run_phase "Phase 4.12: Zero-disc steering" \
    phase 4.12 --parallel 4 --model google/gemma-2-9b

run_phase "Phase 4.14: Statistical significance" \
    phase 4.14 --model google/gemma-2-9b

run_phase "Phase 4.16: Difficulty-stratified steering" \
    phase 4.16 --model google/gemma-2-9b

run_phase "Phase 6.3: Attention pattern analysis" \
    phase 6.3 --model google/gemma-2-9b

# ============================================================
# STAGE 5: Weight Orthogonalization (SAE)
# ============================================================
echo ""
echo "############ STAGE 5: Weight Orthogonalization (SAE) ############"

run_phase "Phase 5.3: Weight orthogonalization" \
    phase 5.3 --parallel 4 --model google/gemma-2-9b

run_phase "Phase 5.6: Zero-disc orthogonalization (SAE-only)" \
    phase 5.6 --parallel 4 --model google/gemma-2-9b

run_phase "Phase 5.9: Orthogonalization significance" \
    phase 5.9 --model google/gemma-2-9b

# ============================================================
# STAGE 6: Instruction-Tuned Model (SAE)
# ============================================================
echo ""
echo "############ STAGE 6: Instruction-Tuned Model (SAE) ############"

run_phase "Phase 7.3: Instruct baseline" \
    phase 7.3 --parallel 4 --model google/gemma-2-9b

run_phase "Phase 7.6: Instruct steering" \
    phase 7.6 --parallel 4 --model google/gemma-2-9b

run_phase "Phase 7.7: Instruct zero-disc (SAE-only)" \
    phase 7.7 --parallel 4 --model google/gemma-2-9b

run_phase "Phase 7.9: Universality analysis" \
    phase 7.9 --model google/gemma-2-9b

run_phase "Phase 7.12: Instruct AUROC/F1" \
    phase 7.12 --model google/gemma-2-9b

# ============================================================
# STAGE 7: Selective Steering (SAE)
# ============================================================
echo ""
echo "############ STAGE 7: Selective Steering (SAE) ############"

run_phase "Phase 8.1: Percentile threshold calculator" \
    phase 8.1 --model google/gemma-2-9b

run_phase "Phase 8.2: Percentile threshold optimizer" \
    phase 8.2 --parallel 4 --model google/gemma-2-9b

run_phase "Phase 8.3: Selective steering" \
    phase 8.3 --parallel 4 --model google/gemma-2-9b

run_phase "Phase 8.7: Threshold search visualization" \
    phase 8.7 --model google/gemma-2-9b

# ============================================================
# STAGE 8: Probe Prediction (probe_logreg)
# ============================================================
echo ""
echo "############ STAGE 8: Probe Prediction ############"

run_phase "Phase 3.8: Probe AUROC/F1 (logreg)" \
    phase 3.8 --direction-source probe_logreg --model google/gemma-2-9b

run_phase "Phase 3.10: Probe temperature AUROC (logreg)" \
    phase 3.10 --direction-source probe_logreg --model google/gemma-2-9b

# ============================================================
# STAGE 9: Probe Steering (probe_mass_mean)
# ============================================================
echo ""
echo "############ STAGE 9: Probe Steering ############"

run_phase "Phase 4.5: Probe coefficient search (mass_mean)" \
    phase 4.5 --direction-source probe_mass_mean --parallel 4 --model google/gemma-2-9b

run_phase "Phase 4.6: Probe golden section (mass_mean)" \
    phase 4.6 --direction-source probe_mass_mean --parallel 4 --model google/gemma-2-9b

run_phase "Phase 4.8: Probe steering effect (mass_mean)" \
    phase 4.8 --direction-source probe_mass_mean --parallel 4 --model google/gemma-2-9b

run_phase "Phase 6.3: Probe attention pattern analysis (mass_mean)" \
    phase 6.3 --direction-source probe_mass_mean --model google/gemma-2-9b

# ============================================================
# STAGE 10: Probe Orthogonalization
# ============================================================
echo ""
echo "############ STAGE 10: Probe Orthogonalization ############"

run_phase "Phase 5.3: Probe weight orthogonalization (mass_mean)" \
    phase 5.3 --direction-source probe_mass_mean --parallel 4 --model google/gemma-2-9b

# ============================================================
# STAGE 11: Probe Instruction-Tuned
# ============================================================
echo ""
echo "############ STAGE 11: Probe Instruction-Tuned ############"

run_phase "Phase 7.6: Probe instruct steering (mass_mean)" \
    phase 7.6 --direction-source probe_mass_mean --parallel 4 --model google/gemma-2-9b

run_phase "Phase 7.12: Probe instruct AUROC/F1 (logreg)" \
    phase 7.12 --direction-source probe_logreg --model google/gemma-2-9b

# ============================================================
# STAGE 12: Probe Selective Steering
# ============================================================
echo ""
echo "############ STAGE 12: Probe Selective Steering ############"

run_phase "Phase 8.1: Probe threshold calculator (logreg)" \
    phase 8.1 --direction-source probe_logreg --model google/gemma-2-9b

run_phase "Phase 8.2: Probe threshold optimizer (logreg)" \
    phase 8.2 --direction-source probe_logreg --parallel 4 --model google/gemma-2-9b

run_phase "Phase 8.3: Probe selective steering (logreg)" \
    phase 8.3 --direction-source probe_logreg --parallel 4 --model google/gemma-2-9b

# ============================================================
# STAGE 13: Error Type Summary
# ============================================================
echo ""
echo "############ STAGE 13: Error Type Summary ############"

run_phase "Phase 9.5: Error type summary" \
    phase 9.5 --model google/gemma-2-9b

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
echo "Full Pipeline Complete!"
echo "Finished: $(date)"
echo "Total elapsed: ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "============================================================"
