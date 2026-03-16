#!/bin/bash
# Phase 9.5 + 10.5 (SAE + probe) for Gemma-2-9B and LLAMA-3.1-8B.
#
# Prereqs (all must exist):
#   Gemma-9B SAE : phase4_9_gemma9b, phase5_3_gemma9b, phase8_1_gemma9b, phase8_2_gemma9b
#   Gemma-9B probe: phase4_8_gemma9b_probe, phase5_3_gemma9b_probe, phase8_1_gemma9b_probe, phase8_2_gemma9b_probe
#   LLAMA SAE    : phase4_9_llama, phase5_3_llama, phase8_1_llama, phase8_2_llama
#   LLAMA probe  : phase4_8_llama_probe, phase5_3_llama_probe, phase8_1_llama_probe, phase8_2_llama_probe
#
# Usage:
#   screen -dmS phase95_105_other bash scripts/run_phase95_105_other_models.sh

set -e

LOG_FILE="scripts/phase95_105_other_models.log"
exec > >(tee -a "$LOG_FILE") 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

cd /home/kriz.tahimic/sae-code-correctness

echo "============================================================"
echo "Phase 9.5 + 10.5 — Gemma-9B + LLAMA (SAE + probe)"
echo "Started: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "============================================================"

# ── helpers ────────────────────────────────────────────────────────────────────

run_phase() {
    local desc="$1"; shift
    echo ""
    echo "=============================="
    echo "$desc"
    echo "Started: $(date)"
    echo "=============================="
    local start=$SECONDS
    python3 run.py "$@"
    echo "Finished: $(date) ($(( SECONDS - start ))s)"
}

# Patch phase10_5 summary with the real phase9_5 preservation_rate, then re-viz.
# Args: <phase9_5_summary_path> <phase10_5_summary_path> <model_flag...>
patch_and_reviz() {
    local p9_summary="$1"
    local p10_summary="$2"
    shift 2
    local reviz_args=("$@")   # remaining args passed to --viz-only run

    echo ""
    echo "--- Patching Phase 10.5 summary with real Phase 9.5 preservation rate ---"
    python3 - "$p9_summary" "$p10_summary" <<'EOF'
import json, sys, pathlib

p9_path  = pathlib.Path(sys.argv[1])
p10_path = pathlib.Path(sys.argv[2])

if not p9_path.exists():
    print(f"WARNING: {p9_path} not found — skipping patch"); sys.exit(0)
if not p10_path.exists():
    print(f"WARNING: {p10_path} not found — skipping patch"); sys.exit(0)

with open(p9_path)  as f: s9  = json.load(f)
with open(p10_path) as f: s10 = json.load(f)

pres_rate = s9.get("preservation_experiment", {}).get("preservation_rate")
if pres_rate is None:
    print("WARNING: preservation_rate not found in Phase 9.5 summary — skipping"); sys.exit(0)

s10.setdefault("comparison_rates", {}).setdefault("phase9_5", {})["preservation_rate"] = pres_rate
with open(p10_path, "w") as f: json.dump(s10, f, indent=2)
print(f"Patched phase9_5.preservation_rate = {pres_rate:.2f}% → {p10_path}")
EOF

    echo "--- Re-viz Phase 10.5 ---"
    python3 run.py "${reviz_args[@]}" --viz-only
}

# ══════════════════════════════════════════════════════════════════════════════
# GEMMA-2-9B
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "############################################################"
echo "GEMMA-2-9B"
echo "############################################################"

run_phase "Phase 9.5 SAE — Gemma-9B" \
    phase 9.5 --model google/gemma-2-9b --parallel 4

run_phase "Phase 10.5 SAE — Gemma-9B" \
    phase 10.5 --model google/gemma-2-9b --parallel 4

patch_and_reviz \
    data/phase9_5_gemma9b/phase_9_5_summary.json \
    data/phase10_5_gemma9b/phase_10_5_summary.json \
    phase 10.5 --model google/gemma-2-9b

run_phase "Phase 9.5 probe — Gemma-9B" \
    phase 9.5 --model google/gemma-2-9b --direction-source probe_mass_mean --parallel 4

run_phase "Phase 10.5 probe — Gemma-9B" \
    phase 10.5 --model google/gemma-2-9b --direction-source probe_mass_mean --parallel 4

patch_and_reviz \
    data/phase9_5_gemma9b_probe/phase_9_5_summary.json \
    data/phase10_5_gemma9b_probe/phase_10_5_summary.json \
    phase 10.5 --model google/gemma-2-9b --direction-source probe_mass_mean

# ══════════════════════════════════════════════════════════════════════════════
# LLAMA-3.1-8B
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "############################################################"
echo "LLAMA-3.1-8B"
echo "############################################################"

run_phase "Phase 9.5 SAE — LLAMA" \
    phase 9.5 --model meta-llama/Llama-3.1-8B --parallel 4

run_phase "Phase 10.5 SAE — LLAMA" \
    phase 10.5 --model meta-llama/Llama-3.1-8B --parallel 4

patch_and_reviz \
    data/phase9_5_llama/phase_9_5_summary.json \
    data/phase10_5_llama/phase_10_5_summary.json \
    phase 10.5 --model meta-llama/Llama-3.1-8B

run_phase "Phase 9.5 probe — LLAMA" \
    phase 9.5 --model meta-llama/Llama-3.1-8B --direction-source probe_mass_mean --parallel 4

run_phase "Phase 10.5 probe — LLAMA" \
    phase 10.5 --model meta-llama/Llama-3.1-8B --direction-source probe_mass_mean --parallel 4

patch_and_reviz \
    data/phase9_5_llama_probe/phase_9_5_summary.json \
    data/phase10_5_llama_probe/phase_10_5_summary.json \
    phase 10.5 --model meta-llama/Llama-3.1-8B --direction-source probe_mass_mean

# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "============================================================"
echo "Phase 9.5 + 10.5 other models complete! $(date)"
echo "============================================================"
