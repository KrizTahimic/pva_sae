# Developmental Interpretability Experiments for Code Correctness

## Overview

This document outlines additional experiments to extend my thesis "Mechanistic Interpretability of Code Correctness in LLMs via Sparse Autoencoders" with a developmental interpretability perspective — understanding **when** and **how** code correctness features emerge during training.

---

## Core Research Question

> How do code correctness features develop over the course of training? Do they emerge suddenly at phase transitions, or gradually? Does the influence between correct and incorrect code samples change over time?

---

## Proposed Experiments

### Experiment 2: Feature Emergence Timeline

**Goal:** Track when code correctness directions become predictive during training

**Method:**
1. Use Pythia checkpoints (e.g., Pythia-160M at steps 1k, 5k, 10k, 30k, 70k, 143k)
2. For each checkpoint:
   - Train an SAE with same architecture
   - Search for code correctness directions using same methodology as main paper
   - Measure prediction accuracy of the direction
3. Plot: Prediction accuracy vs training step

**Expected outcome:** Either gradual emergence or sharp phase transition

**Related paper:** Wang et al. 2025b (rLLC tracking)

---

### Experiment 1: Influence Dynamics Between Code Samples

**Goal:** Test if influence between correct/incorrect code changes non-monotonically over training (sign flips, peaks at transitions)

**Method:**
1. Create dataset:
   - 50 correct code samples
   - 50 incorrect code samples (same problems, buggy solutions)
2. For each Pythia checkpoint:
   - Run RMSProp-SGLD sampling around checkpoint
   - Compute BIF matrix between all sample pairs
   - Aggregate: BIF(correct→correct), BIF(correct→incorrect), BIF(incorrect→incorrect)
3. Plot three curves over training steps

**What to look for:**
- Sign flips (positive → negative influence or vice versa)
- Peaks at specific checkpoints (indicates phase transition)
- Non-monotonic behavior

**Related paper:** Lee et al. 2025 (Influence Dynamics and Stagewise Data Attribution)

---

### Experiment 3: Susceptibility to Code Correctness

**Goal:** Measure which model components respond to shifts toward correct vs incorrect code

**Method:**
1. Define perturbation directions:
   - $p_{correct}$: distribution over correct code
   - $p_{incorrect}$: distribution over incorrect code
2. For each component (attention head, MLP layer, or SAE feature):
   - Compute susceptibility to each perturbation
   - $\chi = -\text{Cov}(\text{component observable}, \text{loss on perturbed data})$
3. Build susceptibility matrix: components × perturbation directions
4. Identify components that respond oppositely to correct vs incorrect

**Can be done:** At single checkpoint (structural) or across checkpoints (developmental)

**Related paper:** Baker et al. 2025 (Susceptibilities)

---

### Experiment 4: Developmental Visualization

**Goal:** Visualize emergence of code-related structure over training

**Method:**
1. Compute susceptibility matrix at each checkpoint
2. Apply UMAP to visualize component organization
3. Create animation or grid showing structural development

**Related paper:** Wang et al. 2025a (Embryology of a Language Model)

---

## Implementation Details

### Pythia Checkpoints

```python
from transformers import AutoModelForCausalLM

checkpoints = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 
               1000, 2000, 4000, 8000, 16000, 32000, 
               64000, 100000, 143000]

for step in checkpoints:
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m",
        revision=f"step{step}"
    )
```

### BIF Estimation (from Influence Dynamics paper)

Core equation:
$$\text{BIF}(z_i, \ell_j) = -\text{Cov}_{p(w|D)}(\ell_i(w), \ell_j(w))$$

Estimated via RMSProp-SGLD:
$$w_{s+1} = w_s - \frac{\hat{\epsilon}_s}{2}\left[\sum_{i} \nabla_w \ell_i(w_s) + \gamma(w_s - w^*)\right] + \mathcal{N}(0, \hat{\epsilon}_s)$$

Key hyperparameters (from paper):
- Step size $\epsilon$: 1e-6 to 1e-3
- Inverse temperature $\beta$: 100 to 10000  
- Localization strength $\gamma$: 100 to 1M
- Chain length: 200-1000 steps
- Number of chains: 4-8

### Susceptibility Estimation

$$\chi_{ij} = -\text{Cov}(\phi_i(w), \ell_j(w))$$

Where:
- $\phi_i(w)$ = observable for component $i$ (e.g., output norm of head)
- $\ell_j(w)$ = loss on data distribution $j$ (e.g., correct code)

---

## Dataset Requirements

### Code Samples for Influence/Susceptibility

Need paired correct/incorrect code:

| Type | Description | Source |
|------|-------------|--------|
| Correct | Working solutions to coding problems | HumanEval, MBPP |
| Incorrect | Buggy solutions to same problems | Mutated correct solutions, or LLM failures |

Minimum: 50 pairs
Recommended: 100-200 pairs for stable covariance estimates

---

## Expected Results

### If code correctness emerges at a phase transition:
- Sharp jump in prediction accuracy at specific checkpoint
- Peak in BIF(correct→correct) at same checkpoint
- Non-monotonic influence curves

### If code correctness emerges gradually:
- Smooth increase in prediction accuracy
- Monotonically decreasing influence (as loss decreases)
- No sharp peaks

---

## Priority Order

1. **Experiment 1 (Influence Dynamics)** — Most novel, directly tests developmental hypothesis
2. **Experiment 2 (Feature Emergence)** — Straightforward extension of existing work
3. **Experiment 3 (Susceptibility)** — Can do at single checkpoint first
4. **Experiment 4 (Visualization)** — Nice to have, depends on results from 1-3

---

## High-Relevance Papers (Detailed)

### Lee et al. 2025 — Influence Dynamics and Stagewise Data Attribution
- **arXiv:** 2510.12071
- **Key idea:** Influence between samples changes over training, peaks at phase transitions
- **Method:** BIF estimated via RMSProp-SGLD across checkpoints
- **Use for:** Experiment 1

### Wang et al. 2025b — Differentiation and Specialization of Attention Heads
- **arXiv:** 2410.02984
- **Venue:** ICLR 2025 Spotlight
- **Key idea:** Refined LLC tracks individual component complexity over training
- **Method:** 
  - Weight-refined LLC: complexity of specific head
  - Data-refined LLC: complexity on specific data subset
- **Use for:** Tracking which heads specialize to code

### Baker et al. 2025 — Structural Inference via Susceptibilities
- **arXiv:** 2504.18274
- **Key idea:** Perturb data distribution, measure component responses
- **Method:** Susceptibility matrix from SGLD covariances
- **Use for:** Experiment 3

---

## Medium-Relevance Papers (For Future Exploration)

### Wang et al. 2025a — Embryology of a Language Model
- **arXiv:** 2508.00331
- **Key idea:** UMAP on susceptibility matrix reveals "body plan" emergence
- **Discovers:** Novel structures like "spacing fin" for counting spaces
- **Potential use:** Visualize code structure emergence

### Carroll et al. 2025 — Dynamics of Transient Structure
- **arXiv:** 2501.17745
- **Key idea:** Models can learn general solution first, then specialize (transient ridge phenomenon)
- **Method:** Joint trajectory PCA + LLC tracking
- **Potential use:** Does model learn "general programming" before "correctness"?

### Urdshals & Urdshals 2025 — Structure Development in List-Sorting
- **arXiv:** 2501.18666
- **Key idea:** Vocabulary-splitting and copy-suppression modes emerge in sorting task
- **Method:** LLC + mechanistic analysis of attention heads
- **Potential use:** Methodology template for algorithmic task interpretability

---

## Low-Relevance Papers (Background Reading)

### Chen et al. 2023 — Phase Transitions in Toy Model of Superposition
- **arXiv:** 2310.06301
- **Key idea:** k-gon critical points determine phase transitions in TMS
- **Very theoretical:** Primarily useful for understanding SLT foundations
- **Why low relevance:** Toy model only, no language model experiments

---

## Tools and Resources

### DevInterp Python Package
```bash
pip install devinterp
```
- LLC estimation
- SGLD sampling
- Documentation: https://devinterp.com

### Timaeus Research
- Website: https://timaeus.co/research
- Papers: https://devinterp.com/research

### Key References for SLT Background
- Watanabe 2009 — "Algebraic Geometry and Statistical Learning Theory"
- Watanabe 2018 — "Mathematical Theory of Bayesian Statistics"
- Lau et al. 2025 — "The Local Learning Coefficient" (foundational LLC paper)

---

## Timeline Estimate

| Experiment | Compute | Analysis | Total |
|------------|---------|----------|-------|
| Exp 1: Influence Dynamics | 3-5 days | 2 days | 5-7 days |
| Exp 2: Feature Emergence | 2-3 days | 1 day | 3-4 days |
| Exp 3: Susceptibility | 1-2 days | 1 day | 2-3 days |
| Exp 4: Visualization | 1 day | 1 day | 2 days |

**Total estimated time:** 2-3 weeks

---

## Notes

- Pythia models are small enough to run SGLD on consumer GPU (RTX 3090 sufficient for 160M)
- BIF estimation is embarrassingly parallel across checkpoints
- Start with fewer checkpoints (5-6) to validate methodology, then densify if interesting patterns emerge