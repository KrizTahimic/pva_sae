#!/bin/bash
# Full LLAMA-3.1-8B pipeline (SAE + Probe) — all phases from scratch.
#
# Prerequisites (do these BEFORE running):
#   1. Clear __pycache__:  find . -type d -name __pycache__ -exec rm -rf {} +
#   2. Ensure Gemma-2B default data exists (Phase 0/0.1 are dataset-level, shared)
#   3. LLAMA outputs go to *_llama/ dirs, so no conflict with Gemma data
#   4. HuggingFace login:  huggingface-cli login  (LLAMA requires gated access)
#
# Config: meta-llama/Llama-3.1-8B + mbpp (passed via --model flag)
# SAE: LlamaScope 8x (fnlp/Llama3_1-8B-Base-LXR-8x), 32k latents, TopK
#
# Usage:
#   screen -S llama8b
#   bash scripts/run_llama8b_clean_rerun.sh
#
# Note: LLAMA-3.1-8B (8B params, 32 layers, 32k SAE) is significantly
# larger than Gemma-2B (2B params, 26 layers, 16k SAE). Expect longer runtimes.
# Total: 53 phase runs (6 foundation + 35 SAE + 12 probe)

set -e

LOG_FILE="scripts/llama8b_clean_rerun.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

MODEL="meta-llama/Llama-3.1-8B"
INSTRUCT_MODEL="meta-llama/Llama-3.1-8B-Instruct"

PIPELINE_START=$(date +%s)

echo "============================================================"
echo "LLAMA-3.1-8B Clean Full Rerun"
echo "Started: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "Model: $MODEL"
echo "SAE: LlamaScope 8x (32k latents, TopK)"
echo "============================================================"

# ============================================================
# STAGE -1: Download Models & SAE from HuggingFace
# ============================================================
echo ""
echo "############ Downloading Models & SAE ############"

echo "Downloading $MODEL..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.1-8B')
print('Base model downloaded.')
"

echo "Downloading $INSTRUCT_MODEL..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.1-8B-Instruct')
print('Instruct model downloaded.')
"

echo "Downloading LlamaScope SAE (fnlp/Llama3_1-8B-Base-LXR-8x)..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('fnlp/Llama3_1-8B-Base-LXR-8x')
print('LlamaScope SAE downloaded.')
"

echo "All downloads complete."

# Patch config.py for LLAMA instruct model
# Phases 7.x read instruct model from config fields, not from --model flag
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
# STAGE 1: Gemma 2B Temperature Sweep
# ============================================================
echo ""
echo "############ STAGE 1: Gemma 2B Temperature Sweep ############"

run_phase "Phase 3.5: Gemma 2B temperature robustness" \
    phase 3.5 --parallel 4

# ============================================================
# STAGE 2: Data Preparation
# ============================================================
echo ""
echo "############ STAGE 2: Data Preparation ############"

run_phase "Phase 0: Difficulty analysis" \
    phase 0 --model "$MODEL"

run_phase "Phase 0.1: Problem splitting" \
    phase 0.1 --model "$MODEL"

# ============================================================
# STAGE 3: Code Generation + Activations
# ============================================================
echo ""
echo "############ STAGE 3: Code Generation ############"

run_phase "Phase 1: Code generation + activation extraction" \
    phase 1 --model "$MODEL" --parallel 4

# ============================================================
# STAGE 4: Feature Discovery (SAE + Probe)
# ============================================================
echo ""
echo "############ STAGE 4: Feature Discovery ############"

run_phase "Phase 2.2: Pile activation caching" \
    phase 2.2 --model "$MODEL" --parallel 4

run_phase "Phase 2.3: Pile SAE frequency computation" \
    phase 2.3 --model "$MODEL"

run_phase "Phase 2.5: SAE analysis + pile filtering" \
    phase 2.5 --model "$MODEL"

run_phase "Phase 2.6: Probe training (logreg + mass_mean)" \
    phase 2.6 --model "$MODEL"

run_phase "Phase 2.10: T-statistic latent selection" \
    phase 2.10 --model "$MODEL"

run_phase "Phase 2.11: Direction similarity analysis" \
    phase 2.11 --model "$MODEL"

run_phase "Phase 2.13: Threshold sensitivity analysis" \
    phase 2.13 --model "$MODEL"

run_phase "Phase 2.15: Layer-wise analysis visualization" \
    phase 2.15 --model "$MODEL"

run_phase "Phase 2.20: Latent landscape visualization" \
    phase 2.20 --model "$MODEL"

# ============================================================
# STAGE 5: Statistical Validation (SAE)
# ============================================================
echo ""
echo "############ STAGE 5: Statistical Validation (SAE) ############"

run_phase "Phase 3.5: Temperature robustness" \
    phase 3.5 --model "$MODEL" --parallel 4

run_phase "Phase 3.6: Hyperparameter baseline" \
    phase 3.6 --model "$MODEL" --parallel 4

run_phase "Phase 3.8: AUROC/F1 evaluation" \
    phase 3.8 --model "$MODEL"

run_phase "Phase 3.10: Temperature-based AUROC" \
    phase 3.10 --model "$MODEL"

run_phase "Phase 3.11: Temperature trends visualization" \
    phase 3.11 --model "$MODEL"

run_phase "Phase 3.12: Difficulty-based AUROC" \
    phase 3.12 --model "$MODEL"

# ============================================================
# STAGE 6: Steering Pipeline (SAE)
# ============================================================
echo ""
echo "############ STAGE 6: Steering Pipeline (SAE) ############"

run_phase "Phase 4.5: Coefficient grid search" \
    phase 4.5 --model "$MODEL" --parallel 4

run_phase "Phase 4.6: Golden section refinement" \
    phase 4.6 --model "$MODEL" --parallel 4

run_phase "Phase 4.7: Coefficient visualization" \
    phase 4.7 --model "$MODEL"

run_phase "Phase 4.8: Steering effect analysis" \
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

run_phase "Phase 6.3: Attention pattern analysis" \
    phase 6.3 --model "$MODEL"

# ============================================================
# STAGE 7: Weight Orthogonalization (SAE)
# ============================================================
echo ""
echo "############ STAGE 7: Weight Orthogonalization (SAE) ############"

run_phase "Phase 5.3: Weight orthogonalization" \
    phase 5.3 --model "$MODEL" --parallel 4

run_phase "Phase 5.6: Zero-disc orthogonalization (SAE-only)" \
    phase 5.6 --model "$MODEL" --parallel 4

run_phase "Phase 5.9: Orthogonalization significance" \
    phase 5.9 --model "$MODEL"

# ============================================================
# STAGE 8: Instruction-Tuned Model (SAE)
# ============================================================
echo ""
echo "############ STAGE 8: Instruction-Tuned Model (SAE) ############"
echo "Using instruct model: $INSTRUCT_MODEL"

run_phase "Phase 7.3: Instruct baseline" \
    phase 7.3 --model "$MODEL" --parallel 4

run_phase "Phase 7.6: Instruct steering" \
    phase 7.6 --model "$MODEL" --parallel 4

run_phase "Phase 7.7: Instruct zero-disc (SAE-only)" \
    phase 7.7 --model "$MODEL" --parallel 4

run_phase "Phase 7.9: Universality analysis" \
    phase 7.9 --model "$MODEL"

run_phase "Phase 7.12: Instruct AUROC/F1" \
    phase 7.12 --model "$MODEL"

# ============================================================
# STAGE 9: Selective Steering (SAE)
# ============================================================
echo ""
echo "############ STAGE 9: Selective Steering (SAE) ############"

run_phase "Phase 8.1: Percentile threshold calculator" \
    phase 8.1 --model "$MODEL"

run_phase "Phase 8.2: Percentile threshold optimizer" \
    phase 8.2 --model "$MODEL" --parallel 4

run_phase "Phase 8.3: Selective steering" \
    phase 8.3 --model "$MODEL" --parallel 4

run_phase "Phase 8.7: Threshold search visualization" \
    phase 8.7 --model "$MODEL"

# ============================================================
# STAGE 10: Probe Prediction (probe_logreg)
# ============================================================
echo ""
echo "############ STAGE 10: Probe Prediction ############"

run_phase "Phase 3.8: Probe AUROC/F1 (logreg)" \
    phase 3.8 --model "$MODEL" --direction-source probe_logreg

run_phase "Phase 3.10: Probe temperature AUROC (logreg)" \
    phase 3.10 --model "$MODEL" --direction-source probe_logreg

# ============================================================
# STAGE 11: Probe Steering (probe_mass_mean)
# ============================================================
echo ""
echo "############ STAGE 11: Probe Steering ############"

run_phase "Phase 4.5: Probe coefficient search (mass_mean)" \
    phase 4.5 --model "$MODEL" --direction-source probe_mass_mean --parallel 4

run_phase "Phase 4.6: Probe golden section (mass_mean)" \
    phase 4.6 --model "$MODEL" --direction-source probe_mass_mean --parallel 4

run_phase "Phase 4.8: Probe steering effect (mass_mean)" \
    phase 4.8 --model "$MODEL" --direction-source probe_mass_mean --parallel 4

run_phase "Phase 6.3: Probe attention pattern analysis (mass_mean)" \
    phase 6.3 --model "$MODEL" --direction-source probe_mass_mean

# ============================================================
# STAGE 12: Probe Orthogonalization
# ============================================================
echo ""
echo "############ STAGE 12: Probe Orthogonalization ############"

run_phase "Phase 5.3: Probe weight orthogonalization (mass_mean)" \
    phase 5.3 --model "$MODEL" --direction-source probe_mass_mean --parallel 4

# ============================================================
# STAGE 13: Probe Instruction-Tuned
# ============================================================
echo ""
echo "############ STAGE 13: Probe Instruction-Tuned ############"

run_phase "Phase 7.6: Probe instruct steering (mass_mean)" \
    phase 7.6 --model "$MODEL" --direction-source probe_mass_mean --parallel 4

run_phase "Phase 7.12: Probe instruct AUROC/F1 (logreg)" \
    phase 7.12 --model "$MODEL" --direction-source probe_logreg

# ============================================================
# STAGE 14: Probe Selective Steering
# ============================================================
echo ""
echo "############ STAGE 14: Probe Selective Steering ############"

run_phase "Phase 8.1: Probe threshold calculator (logreg)" \
    phase 8.1 --model "$MODEL" --direction-source probe_logreg

run_phase "Phase 8.2: Probe threshold optimizer (logreg)" \
    phase 8.2 --model "$MODEL" --direction-source probe_logreg --parallel 4

run_phase "Phase 8.3: Probe selective steering (logreg)" \
    phase 8.3 --model "$MODEL" --direction-source probe_logreg --parallel 4

# ============================================================
# STAGE 15: Error Type Summary
# ============================================================
echo ""
echo "############ STAGE 15: Error Type Summary ############"

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
echo "Full LLAMA-3.1-8B Pipeline Complete!"
echo "Finished: $(date)"
echo "Total elapsed: ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "============================================================"
