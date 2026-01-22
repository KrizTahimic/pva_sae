---
description: Run standard testing routine for a phase (checkpointing, parallel, multi-model)
argument-hint: "<phase_number> [--model llama|gemma] [--end N]"
allowed-tools: Bash(*), Read(*), Grep(*)
---

Run the standard testing routine for a phase. This verifies:
1. Phase runs without errors
2. Dependencies are auto-discovered
3. Cross-run checkpointing works (re-run skips processed tasks)
4. Parallel execution works (4 GPUs)
5. Output files are correct

## Arguments
- `phase_number`: Required. The phase to test (e.g., "1", "2.5", "4.8")
- `--model`: Optional. "llama" or "gemma" (default: current config)
- `--end N`: Optional. Test subset size (default: 39)

## Testing Steps

### Step 1: Run subset test
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sae_cc
python3 run.py phase {N} --parallel 4 --end {END}
```
Verify: No errors, output directory created, phase_output.json exists

### Step 2: Test checkpointing (re-run same command)
```bash
python3 run.py phase {N} --parallel 4 --end {END}
```
Verify: Completes in seconds, logs show "checkpointed" or "skipping already processed"

### Step 3: Check outputs
- Verify summary JSON has expected metrics
- Verify results in expected ranges:
  - Pass rate: 25-35%
  - Correction rate: 5-40%
  - Corruption rate: 5-30%
  - AUROC: 0.6-0.9

### Step 4 (if --model specified): Multi-model test
Edit config.py to switch model, then re-run step 1.

User request: $ARGUMENTS
