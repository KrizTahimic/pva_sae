# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## 🚨 CRITICAL CHECKLIST - Read Before Every Task

**YOU MUST follow these rules. Violations waste hours of GPU time.**

1. **NO backward compatibility** - Make clean breaks. Delete old code. Update all call sites.
   ```python
   # ❌ WRONG                          # ✅ CORRECT
   if manifest.exists():               if not manifest.exists():
       return load_new(manifest)           raise FileNotFoundError("Run phase X first")
   return legacy_fallback(path)        return load_new(manifest)
   ```

2. **Read before editing** - Never propose changes to code you haven't read first.

3. **Ask before implementing** - Get explicit user approval before writing code.

4. **Activate conda first**:
   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh && conda activate sae_cc
   ```

5. **Use screen for long tasks** - Provide screen instructions for user to run manually. Do NOT execute screen commands via Claude Code.

---

## Project Context

### What This Project Does

SAE-based analysis of code correctness representations in LLMs. Uses Gemma 2 models, GemmaScope SAEs, and MBPP/HumanEval datasets to:
- Generate code solutions and classify as correct/incorrect
- Identify SAE latent directions encoding correctness
- Validate via AUROC/F1 and causal steering interventions

### Why 30+ Phases?

**Resource-constrained thesis project** - we don't have abundant compute.

| If run fails... | Research Lab | This Project |
|-----------------|--------------|--------------|
| | Re-run, no problem | Lost hours of GPU time |

Each phase is a **checkpoint boundary**. If Phase 4.5 fails, you don't lose Phases 1-4.

Key infrastructure:
- `--start N --end M` flags: Test on subset before committing to 6-hour run
- Checkpoints every 50 records: Resume after SSH disconnect
- Auto-discovery: Phases find outputs from previous phases automatically

---

## Running Phases

### Basic Command

```bash
python3 run.py phase {N} [OPTIONS]
```

### Common Phases

| Category | Phases | Purpose |
|----------|--------|---------|
| Data prep | 0, 0.1 | Difficulty analysis, problem splitting |
| Feature discovery | 1, 2.2, 2.5, 2.10 | Dataset generation, SAE analysis |
| Validation | 3.5, 3.8 | Temperature robustness, AUROC/F1 |
| Steering | 4.5, 4.8, 4.14 | Coefficient selection, effect analysis |
| Model comparison | 7.3, 7.12 | Instruction-tuned evaluation |
| Selective | 8.3 | Threshold-based steering |

### Key Options

```bash
--start N --end M              # Process subset (for testing)
--viz-only                     # Regenerate plots without recomputing (seconds vs hours)
--correction-only              # Only correction experiments (phases 4.5, 4.6, 4.8)
--parallel N                   # Multi-GPU parallelization (distribute tasks across N GPUs)
--direction-source SOURCE      # sae (default), probe_logreg, or probe_mass_mean
```

### Multi-GPU Parallelization

Phases with `model.generate()` support `--parallel N` for ~3-4x speedup:

```bash
# Run Phase 1 across 4 GPUs
python3 run.py phase 1 --parallel 4

# Test with subset first
python3 run.py phase 1 --parallel 4 --start 0 --end 40
```

**How it works:**
- Tasks distributed round-robin: GPU 0 gets tasks [0,4,8...], GPU 1 gets [1,5,9...]
- Each GPU saves `results_gpu{N}.parquet`
- Orchestrator merges results after completion

**Supported phases:** 1, 3.5, 3.6, 4.5, 4.6, 4.8, 4.12, 5.3, 5.6, 7.3, 7.6, 8.2, 8.3

See `common/parallel_runner.py` for implementation details.

### Checkpointing

All generation phases checkpoint every 50 records. If interrupted, re-run the same command - it auto-resumes.

---

## Configuration

### Switching Models/Datasets

Edit `common/config.py` directly. No CLI arguments needed.

```python
# common/config.py
dataset_name: str = "humaneval"  # Options: "mbpp", "humaneval"
model_name: str = "google/gemma-2-2b"  # See supported models below
```

### Supported Models

| Model | SAE |
|-------|-----|
| `google/gemma-2-2b` | GemmaScope 16k |
| `google/gemma-2-2b-it` | GemmaScope 16k |
| `google/gemma-2-9b` | GemmaScope 16k |
| `meta-llama/Llama-3.1-8B` | LlamaScope 8x |

### Output Directory Conventions

Directories auto-include model/dataset suffixes:

| Config | Output Directory |
|--------|------------------|
| Gemma-2B + MBPP | `data/phase1_0/` (default) |
| Gemma-9B + MBPP | `data/phase1_0_gemma9b/` |
| LLAMA + MBPP | `data/phase1_0_llama/` |
| Gemma + HumanEval | `data/phase1_0_humaneval/` |

### Path Discovery

```python
from common.phase_discovery import get_phase_output_dir, discover_latest_phase_output

output_dir = Path(get_phase_output_dir("2.2", config))  # For writing
input_path = discover_latest_phase_output("3.5", config=self.config)  # For reading
```

**NEVER** use hardcoded paths like `f"data/phase3_5_{config.dataset_name}"`.

---

## Architecture

### PCDGE Pattern

The fundamental pattern across all phases:
- **P**rompt: Build from MBPP/HumanEval problem
- **C**apture: Extract activations via PyTorch hooks
- **D**ecompose: Apply SAE (GemmaScope JumpReLU)
- **G**enerate: LLM produces code solution
- **E**valuate: Execute tests to classify correct/incorrect

### Key Files

| File | Purpose |
|------|---------|
| `run.py` | Main CLI entry point |
| `common/config.py` | Centralized configuration |
| `common/phase_discovery.py` | Auto-discovery, output paths |
| `common/sae_loader.py` | GemmaScope SAE loading |
| `common/steering_setup.py` | Direction loading (SAE + probe) |
| `common/parallel_runner.py` | Multi-GPU parallelization orchestration |
| `phase2_5_separation_score_analysis/sae_analyzer.py` | Separation score analysis |
| `phase4_8_steering_analysis/steering_effect_analyzer.py` | Steering interventions |

### Direction Source Architecture

**THREE direction sources** for different use cases:

| Source | Method | Best For | CLI Flag |
|--------|--------|----------|----------|
| SAE (Phase 2.5/2.10) | Unsupervised | Discovery | `--direction-source sae` |
| Probe LogReg (Phase 2.6) | Supervised | Detection (AUROC/F1) | `--direction-source probe_logreg` |
| Probe Mass-Mean (Phase 2.6) | Supervised | Steering (correction) | `--direction-source probe_mass_mean` |

**SAE Latent Selection** (legacy, still used):

| Use Case | Source | Metric | Used By |
|----------|--------|--------|---------|
| Validation | Phase 2.10 | t-statistic | Phases 3.x (AUROC/F1) |
| Steering | Phase 2.5 | separation score | Phases 4.x, 5.x, 6.x, 7.x |

**Probe vs SAE:** Different projections of the same linear representation. Probes find directions optimized for the task; SAE finds unsupervised latents with interpretability benefits.

### Activation Extraction

Activations captured at **last prompt token** (position -1) from **residual stream** (`resid_post`) before generation begins.

### Steering Mechanism

```python
# SAE steering
steered_activation = original_activation + coefficient * sae_decoder_direction

# Probe steering (mass-mean)
steered_activation = original_activation + coefficient * probe_direction
```

Both use same hook mechanism; only the direction source differs. Coefficient selection: coarse-to-fine search (Phase 4.5) → golden section refinement (Phase 4.6).

---

## Standards

### SAE Terminology

| Term | Description | Type |
|------|-------------|------|
| `latent_idx` | Integer index into SAE latent space | `int` |
| `latent_direction` | Decoder weight vector (`W_dec[latent_idx]`) | `Tensor [d_model]` |
| `latent_activation` | Scalar activation value | `float` |
| `latent_activations` | Full encoded output | `Tensor [batch, n_latents]` |

**Use "predicting"** (correct-predicting, incorrect-predicting), not "preferring" or "detecting".

**DO NOT USE**: `feature` (use `latent`), `feature_type` (use `latent_type`), `latent_index` (use `latent_idx`)

### Test Outcome Terminology

| Term | Description |
|------|-------------|
| `baseline_passed` | Did unmodified generation pass tests? |
| `steered_correct` | Is steered output correct? |
| `orthogonalized_correct` | Is orthogonalized output correct? |

```python
correction = (baseline_passed == False) & steered_correct      # incorrect → correct
corruption = baseline_passed & (steered_correct == False)      # correct → incorrect
preservation = baseline_passed & steered_correct               # correct → correct
```

### Color Scheme

Constants defined in `common/config.py`:

| Concept | Color | Constant |
|---------|-------|----------|
| Correction / Correct-predicting | Green | `COLOR_CORRECTION` |
| Corruption / Incorrect-predicting | Red | `COLOR_CORRUPTION` |
| Preservation | Gold | `COLOR_PRESERVATION` |

### Tensor Patterns

Use `einops.rearrange` for complex reshapes:

```python
# ✅ GOOD - Shape explicit
from einops import rearrange
steering = rearrange(latent_direction, 'd -> 1 1 d') * coefficient

# ❌ AVOID - Shape not obvious
steering = latent_direction.unsqueeze(0).unsqueeze(0) * coefficient
```

Keep simple operations simple: `x @ self.W_enc`, `activation.squeeze(0)`

### File Formats

**Use `.safetensors`** for all activations and tensors. Do NOT use `.npz` (except GemmaScope SAE params - external format).

```python
from common.tensor_utils import save_tensor, load_tensor
save_tensor(activation, Path("activation.safetensors"))
```

---

## Data Paths

### Directory Structure

```
data/
├── phase0/       # Difficulty mappings
├── phase0_1/     # Split datasets (selection_mbpp.parquet, tuning_mbpp.parquet, analysis_mbpp.parquet)
├── phase1_0/     # Generated code + activations
├── phase2_2/     # Pile activation baseline
├── phase2_5/     # SAE analysis (top_20_latents.json)
├── phase3_8/     # AUROC/F1 evaluation
├── phase4_8/     # Steering effect analysis
└── phase7_12/    # Instruction-tuned comparisons
```

### Auto-Discovery

Later phases find outputs from earlier phases by searching for timestamped files in expected directories. Use `discover_latest_phase_output()`.

---

## Reference

### Hardware Requirements

- GPU: 24GB+ VRAM (for 2B model + SAE)
- CPU RAM: 100GB+ (activation processing)
- Disk: ~50GB (model weights + datasets)

### Adding a New Phase

1. Create `phase{N}_{name}/` directory
2. Add output directory to `common/config.py`
3. Implement runner class following existing patterns
4. Add phase handler to `run.py::main()`

### GemmaScope Integration

- Repo: `google/gemma-scope-2b-pt-res`
- Width: 16k latents
- Activation: JumpReLU (not standard ReLU)
- Layers 0-25 available for Gemma-2B

### Multi-Model Strategy

- **LLAMA**: Gets its own features - run full Phase 1 → 2.5 pipeline
- **HumanEval**: Validation-only - use Gemma+MBPP features, test on HumanEval

### HumanEval Prompt Format

Converted to MBPP-style prompts for experimental consistency (actual test assertions shown, not docstring examples). Differs from standard HumanEval evaluation.

### Reproducibility

- Random seeds fixed (default: 42)
- Temperature=0.0 for deterministic baseline
- Problem splits stratified by difficulty

### GPU Memory Issues

If OOM:
1. Reduce batch sizes in config
2. `--start 0 --end 10` to test subset first
3. `python3 run.py cleanup-gpu --aggressive`

### Maintaining This File

When adding to CLAUDE.md:
1. **Keep it under 400 lines** - Current: ~320 lines. Remove something if adding significantly.
2. **No generic style guides** - Use linters for that. Only project-specific patterns.
3. **Tables over prose** - Easier to scan.
4. **One example max** - Per concept. Not multiple variations.
5. **Reference, don't embed** - Point to `common/config.py` instead of duplicating its contents.

**Structure** (8 sections, maintain this order):
1. CRITICAL CHECKLIST - Rules that prevent wasted GPU hours
2. Project Context - What/why/how
3. Running Phases - Commands and options
4. Configuration - How to switch models/datasets
5. Architecture - PCDGE, key files, latent sources
6. Standards - Terminology, colors, patterns
7. Data Paths - Directory structure, auto-discovery
8. Reference - Hardware, adding phases, maintenance
