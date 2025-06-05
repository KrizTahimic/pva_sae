# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PVA-SAE (Python Value Attribution Sparse Autoencoder) is a thesis research project investigating program validity awareness in language models. The project analyzes how language models internally represent code correctness by:

1. Using Google's Gemma 2 (9B parameters) to generate Python solutions for MBPP problems
2. Classifying solutions as correct (pass@1) or incorrect based on test execution
3. Applying Sparse Autoencoders (SAEs) from GemmaScope to identify latent directions
4. Validating findings through statistical analysis (AUROC, F1) and model steering experiments

The methodology follows three phases: dataset building (50% SAE analysis, 10% hyperparameter tuning, 40% validation), SAE activation analysis using separation scores, and validation through both statistical measures and causal intervention via model steering.

## Memories
- to memorize use python3
- Use `python3 run.py test-gpu` to test GPU setup
- Use `python3 run.py phase 1 --batch-size 16 --num-gpus 3` for multi-GPU generation
- Use Gemma 2 2B for testing: `--model google/gemma-2-2b`
- to memorize this issue so we will prevent this from happening again.
- to memorize

## Project Scale & Technical Details

### 1. Scale & Size
- **MBPP problems**: 974 problems (the full MBPP test set, indices 0-973)
- **Sequence length**: Max 2000 tokens per generation (includes problem description, function signature, and 3 test cases)
- **Layers analyzed**: Configurable per experiment - examples show layer 20, but later after experiments the main codebase will analyze all layers of the residual stream

### 2. Analysis Requirements
- **Token positions**: Default stores only final token (`final_token_only=True`)
- **Components**: Primarily residual stream (resid_pre, resid_mid, resid_post)
- **Re-access**: Activations are extracted on-demand during analysis, not permanently stored

### 3. Computational Resources
- **Memory limits**: 100GB CPU RAM max, 30GB GPU memory per GPU
- **GPU support**: Multi-GPU capability (up to 3 GPUs in parallel)
- **Portability**: Project needs to move between machines (cloud → GPU machines → MacBook)

### 4. Workflow
- **Mixed approach**: One-time dataset building, then iterative experiments
- **Comparison needs**: Yes - steering experiments test multiple coefficients [-1.0 to 1.0]
- **Updates**: Robust checkpointing system - saves every 50 records, autosaves every 100 with 3 versions retained

The project is designed for production-scale research with efficient memory management and multi-GPU support to handle analyzing a 9B parameter model across nearly 1000 code generation problems.

## Key Commands

### Environment Setup
```bash
# Activate conda environment
conda activate pva_sae

# Install dependencies (if not already installed)
pip install -r requirements.txt

# For CUDA support, also install:
pip install accelerate
```