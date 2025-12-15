# Linear Probe vs SAE Sanity Check

Compares linear probe (logistic regression on raw activations) against SAE latent prediction.

## Why This Matters

If a reviewer asks: "Why use SAE instead of a simple linear probe?"

| Result | Interpretation |
|--------|----------------|
| SAE >> Probe | SAE decomposition captures something special |
| SAE ≈ Probe | SAE provides interpretability without losing accuracy |
| SAE << Probe | SAE losing information - need to explain |

## MVP: One Probe, One Layer, Per Model

The sanity check is simple:
- **ONE linear probe per model** (binary: correct vs incorrect)
- **ONE middle layer** (layer 19 by default)
- The probe direction works for both prediction AND steering

## Usage

### MVP: All Models, Middle Layer (Recommended)
```bash
python run_all_models_parallel.py
```

### Single Model
```bash
python run_sanity_check.py --model gemma2b --layer 19
```

### Custom Layers (if needed later)
```bash
python run_all_models_parallel.py --layers 16 19 22
```

## Models Supported

| Model | Phase 1 Dir | HF Model Name |
|-------|-------------|---------------|
| `gemma2b` | phase1_0 | google/gemma-2-2b |
| `gemma2b_it` | phase1_0_it | google/gemma-2-2b-it |
| `gemma9b` | phase1_0_gemma9b | google/gemma-2-9b |
| `llama` | phase1_0_llama | meta-llama/Llama-3.1-8B |

## Requirements

- Phase 1 must be run first (generates activations)
- Minimum 200 samples recommended for meaningful comparison
- SAE weights downloaded (GemmaScope, LlamaScope)

## Output

Compares three methods:
1. **Linear Probe**: Logistic regression on raw activations [d_model]
2. **SAE Latent**: Best single latent from 16k SAE features
3. **Random Direction**: Sanity baseline

Reports AUROC, F1, and declares winner per layer.
