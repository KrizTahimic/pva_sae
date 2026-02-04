---
description: Run a phase with pre-flight checks (GPU memory, dependencies, conda)
argument-hint: "<phase_number> [--parallel N] [--start S --end E] [--direction-source SOURCE]"
allowed-tools: Bash(*), Read(*), Grep(*), Glob(*)
---

Run a phase with safety checks before execution.

## Phases that support --parallel

**Data-parallel:** 1, 3.6, 4.8, 4.12, 5.3, 5.6, 7.3, 7.6, 8.3
**Iterative-parallel:** 3.5, 4.5, 4.6, 8.2

Note: `--parallel` defaults to 4 in run.py. Use `--parallel 1` for sequential execution.

## Step 1: Activate environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sae_cc
```

## Step 2: Pre-flight checks

Run these checks BEFORE executing the phase. If any fail, report the issue and DO NOT run the phase.

### GPU memory check

```bash
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader
```

- For non-parallel runs: Require at least 20GB free on at least one GPU.
- For `--parallel N`: Require N GPUs with 20GB+ free each.
- If adding `--parallel 4` automatically, verify 4 GPUs have 20GB+ free. If not, reduce to available count.

### Dependency check

Check that the phase's input dependencies exist. Use auto-discovery:

```python
python3 -c "
from common.phase_discovery import discover_latest_phase_output
from common.config import Config
config = Config()
# Adjust the prerequisite phase based on what $ARGUMENTS requests
print(discover_latest_phase_output('PREREQ_PHASE', config=config))
"
```

Common dependency chain:
- Phase 1 needs: nothing (generates from scratch)
- Phase 2.x needs: Phase 1 outputs
- Phase 3.x needs: Phase 2.5 or 2.10 outputs
- Phase 4.x needs: Phase 3.x outputs
- Phase 7.x needs: Phase 2.5 outputs
- Phase 8.x needs: Phase 2.6 probe outputs

If the dependency is missing, tell the user which phase they need to run first.

## Step 3: Execute (with screen for parallel runs)

**For phases with `--parallel` (including auto-added):**

Use screen for long-running parallel tasks. Create a screen session named after the phase:

```bash
screen -dmS phase_X bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate sae_cc && python3 run.py phase $ARGUMENTS_WITH_PARALLEL 2>&1 | tee phase_X.log; exec bash'
```

Then tell the user:
- Session started: `screen -r phase_X` to attach
- Log file: `phase_X.log` (can tail with `tail -f phase_X.log`)

**For phases without parallel support:**

Run directly (no screen needed for quick phases):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sae_cc && python3 run.py phase $ARGUMENTS
```

## Step 4: Post-run validation

After the phase completes:
1. Check the exit code (0 = success)
2. List files in the output directory to confirm outputs were created
3. Report a summary: how many files, total size, any obvious issues

For screen sessions, remind the user to check the log file or attach to the session.
