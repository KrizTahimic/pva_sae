#!/bin/bash
# Phase 9.5 + 10.5 with probe_mass_mean direction source.
# Prereqs: phase5_3_probe, phase4_8_probe, phase4_9 (SAE, for steering coeff),
#           phase3_5, phase8_1, phase8_2 must be complete.
#
# Usage:
#   screen -dmS phase95_105_probe bash scripts/run_phase95_105_probe.sh

set -e

LOG_FILE="scripts/phase95_105_probe.log"
exec > >(tee -a "$LOG_FILE") 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae_cc

cd /home/kriz.tahimic/sae-code-correctness

echo "============================================================"
echo "Phase 9.5 + 10.5 — probe_mass_mean"
echo "Started: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "============================================================"

echo ""
echo "=============================="
echo "Phase 9.5: Combined ortho + steering (probe_mass_mean, --parallel 4)"
echo "=============================="
python3 run.py phase 9.5 --direction-source probe_mass_mean --parallel 4

echo ""
echo "=============================="
echo "Patching Phase 10.5 probe summary with real Phase 9.5 preservation rate..."
echo "=============================="
python3 - <<'EOF'
import json, pathlib

summary_path = pathlib.Path("data/phase9_5_probe/phase_9_5_summary.json")
phase10_5_path = pathlib.Path("data/phase10_5_probe/phase_10_5_summary.json")

if not summary_path.exists():
    print(f"ERROR: {summary_path} not found — skipping patch")
    exit(0)

with open(summary_path) as f:
    s9 = json.load(f)

pres_rate = s9.get("preservation_experiment", {}).get("preservation_rate", None)
if pres_rate is None:
    print("ERROR: preservation_experiment.preservation_rate not found in Phase 9.5 summary")
    exit(0)

print(f"Phase 9.5 preservation_rate: {pres_rate:.2f}%")

if not phase10_5_path.exists():
    print(f"Phase 10.5 probe summary not found yet — skipping patch (will be written by Phase 10.5 run)")
    exit(0)

with open(phase10_5_path) as f:
    s10 = json.load(f)

s10.setdefault("comparison_rates", {}).setdefault("phase9_5", {})["preservation_rate"] = pres_rate

with open(phase10_5_path, "w") as f:
    json.dump(s10, f, indent=2)

print(f"Patched {phase10_5_path}")
EOF

echo ""
echo "=============================="
echo "Phase 10.5: Selective ortho + selective steering (probe_mass_mean, --parallel 4)"
echo "=============================="
python3 run.py phase 10.5 --direction-source probe_mass_mean --parallel 4

echo ""
echo "=============================="
echo "Patching Phase 10.5 probe summary with real Phase 9.5 preservation rate and re-viz..."
echo "=============================="
python3 - <<'EOF'
import json, pathlib

summary_path = pathlib.Path("data/phase9_5_probe/phase_9_5_summary.json")
phase10_5_path = pathlib.Path("data/phase10_5_probe/phase_10_5_summary.json")

if not summary_path.exists() or not phase10_5_path.exists():
    print("Skipping patch — one or both summaries missing")
    exit(0)

with open(summary_path) as f:
    s9 = json.load(f)
pres_rate = s9["preservation_experiment"]["preservation_rate"]

with open(phase10_5_path) as f:
    s10 = json.load(f)
s10.setdefault("comparison_rates", {}).setdefault("phase9_5", {})["preservation_rate"] = pres_rate

with open(phase10_5_path, "w") as f:
    json.dump(s10, f, indent=2)

print(f"Patched phase9_5 preservation_rate = {pres_rate:.2f}% in {phase10_5_path}")
EOF

echo ""
echo "=============================="
echo "Phase 10.5: Re-viz with patched rates (probe_mass_mean)"
echo "=============================="
python3 run.py phase 10.5 --direction-source probe_mass_mean --viz-only

echo ""
echo "============================================================"
echo "Phase 9.5 + 10.5 probe complete! $(date)"
echo "============================================================"
