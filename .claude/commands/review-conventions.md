---
description: Review staged/unstaged changes against project conventions before committing
allowed-tools: Bash(git*), Read(*), Grep(*), Glob(*)
---

Review all uncommitted changes against EVERY convention in CLAUDE.md.

## Step 1: Gather changes

```bash
git diff --name-only
git diff --cached --name-only
```

Read each modified/staged file in full.

## Step 2: Check against ALL project conventions

### Critical Checklist (CLAUDE.md § 1)

- **No backward compatibility**: No legacy fallbacks. No `if old_path.exists(): return legacy_load()`. Clean breaks — raise errors if prerequisites are missing.
- **No unused backward-compat artifacts**: No renamed `_vars`, no re-exported types, no `# removed` comments. If something is unused, it must be deleted entirely.

### SAE Terminology (CLAUDE.md § 6 — Standards)

| Required Term | Type | BANNED Alternatives |
|---------------|------|---------------------|
| `latent_idx` | `int` | `feature`, `feature_index`, `latent_index`, `feature_idx` |
| `latent_direction` | `Tensor [d_model]` | `feature_direction`, `direction_vector` |
| `latent_activation` | `float` | `feature_activation` |
| `latent_activations` | `Tensor [batch, n_latents]` | `feature_activations` |
| `latent_type` | str | `feature_type` |

- Must say **"predicting"** (correct-predicting, incorrect-predicting) — NOT "preferring" or "detecting"

### Test Outcome Terminology (CLAUDE.md § 6)

| Required Term | Meaning |
|---------------|---------|
| `baseline_passed` | Did unmodified generation pass tests? |
| `steered_correct` | Is steered output correct? |
| `orthogonalized_correct` | Is orthogonalized output correct? |
| `correction` | `baseline_passed == False` AND `steered_correct` |
| `corruption` | `baseline_passed` AND `steered_correct == False` |
| `preservation` | `baseline_passed` AND `steered_correct` |

### Path Discovery (CLAUDE.md § 4 — Configuration)

- Uses `discover_latest_phase_output()` for reading prior phase outputs
- Uses `get_phase_output_dir()` for writing new outputs
- **NEVER** hardcoded paths like `f"data/phase3_5_{config.dataset_name}"`
- Uses `Path()` objects, not raw string concatenation

### Output Directory Conventions (CLAUDE.md § 4)

- Output dirs auto-include model/dataset suffixes (e.g., `phase1_0_gemma9b/`, `phase1_0_humaneval/`)
- Config changes go in `common/config.py` directly, not via new CLI arguments

### Tensor Patterns (CLAUDE.md § 6)

- Complex reshapes use `einops.rearrange` with explicit shape annotations — NOT `.unsqueeze()` / `.view()` chains
- **Keep simple operations simple**: `x @ self.W_enc`, `activation.squeeze(0)` are fine without einops
- Tensor storage uses `.safetensors` via `save_tensor` / `load_tensor` from `common/tensor_utils` — NOT `.npz` (exception: GemmaScope SAE params which are external format)

### Color Scheme (CLAUDE.md § 6)

- Uses constants from `common/config.py`:
  - `COLOR_CORRECTION` (green) for correction / correct-predicting
  - `COLOR_CORRUPTION` (red) for corruption / incorrect-predicting
  - `COLOR_PRESERVATION` (gold) for preservation
- No hardcoded color strings where these constants should be used

### Architecture Compliance (CLAUDE.md § 5)

- **PCDGE pattern**: New phases should follow Prompt → Capture → Decompose → Generate → Evaluate
- **Direction sources**: SAE (Phase 2.5, separation score) for steering; SAE (Phase 2.10, t-statistic) for validation; Probes (Phase 2.6) for supervised detection/steering
- **Phases 8.2/8.3**: Must use dual-direction internally — logreg for threshold, mass_mean for steering
- **Activation extraction**: At last prompt token (position -1), from residual stream (`resid_post`)
- **Steering formula**: `steered = original + coefficient * direction`

### New Phase Checklist (CLAUDE.md § 8 — Reference)

If the changes add a new phase, verify:
1. Created `phase{N}_{name}/` directory
2. Added output directory to `common/config.py`
3. Implemented runner class following existing patterns
4. Added phase handler to `run.py::main()`

### Reproducibility (CLAUDE.md § 8)

- Random seeds fixed (default: 42) — no unseeded randomness
- Temperature=0.0 for deterministic baseline generation
- Problem splits stratified by difficulty

### Security

- No hardcoded secrets, API keys, or tokens
- No command injection via string formatting in `subprocess` or `os.system`

## Step 3: Report

Organize ALL findings by severity:

- **Critical** — must fix before commit (wrong terminology, hardcoded paths, backward compat fallbacks, security)
- **Warning** — should fix (missing einops, hardcoded colors, missing Path objects)
- **Note** — minor (style, suggestions)

If everything is clean, say so explicitly. Always state how many files were reviewed and how many issues were found.
